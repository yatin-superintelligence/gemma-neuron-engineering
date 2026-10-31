"""Bounded macOS function checking with an explicit fresh-list contract."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
import psutil


WORKER = r'''
import contextlib,copy,io,json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
payload=json.load(sys.stdin)
def equal(a,b):
    if type(a)is not type(b):return False
    if isinstance(b,list):return len(a)==len(b)and all(equal(x,y)for x,y in zip(a,b))
    if isinstance(b,dict):return a.keys()==b.keys()and all(equal(a[k],b[k])for k in b)
    return a==b
def input_lists(value):
    found=[]
    seen=set()
    def visit(item):
        if isinstance(item,list):
            if id(item)in seen:return
            seen.add(id(item))
            found.append(item)
            for child in item:visit(child)
        elif isinstance(item,dict):
            for child in item.values():visit(child)
    visit(value)
    return found
result={'correct':False,'status':'runtime_error','cases_completed':0}
try:
    namespace={'__name__':'candidate'};captured=io.StringIO()
    with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
        exec(compile(payload['source'],'<candidate>','exec'),namespace)
        function=namespace[payload['name']]
        for i,case in enumerate(payload['cases']):
            args=copy.deepcopy(case['args']);before=copy.deepcopy(args)
            original_lists=input_lists(args)
            actual=function(*args)
            if payload['preserve_inputs']and not equal(args,before):
                result.update(status='input_mutation',wrong_case=i);break
            if not equal(actual,case['expected']):
                result.update(status='wrong_answer',wrong_case=i);break
            if payload['require_new_return_list']:
                if type(actual)is not list:
                    result.update(status='return_type_error',wrong_case=i);break
                if any(actual is original for original in original_lists):
                    result.update(status='returned_input_alias',wrong_case=i);break
            result['cases_completed']=i+1
        else:result.update(correct=True,status='passed')
    if captured.getvalue():result.update(correct=False,status='unexpected_IO')
except BaseException as e:
    result.update(status='import_error'if isinstance(e,ImportError)else'runtime_error',error_type=type(e).__name__,error=str(e)[:500])
print(json.dumps(result))
'''


def extract_source(text,name):
    source=text.strip()
    if source.startswith('```'):
        match=re.fullmatch(r'```(?:python|python3|py)?\s*\n([\s\S]*?)\n```',source)
        if not match:raise ValueError('One complete Python/plain fence with no outside prose required')
        source=match[1]
    tree=ast.parse(source)
    if not any(isinstance(n,ast.FunctionDef)and n.name==name for n in tree.body):raise ValueError('Expected complete named function')
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef,ast.Assign,ast.AnnAssign)):continue
        if isinstance(node,ast.Expr)and isinstance(node.value,ast.Constant)and isinstance(node.value.value,str):continue
        raise ValueError('Only module definitions,imports,assignments and docstrings are allowed outside functions')
    banned_modules={'os','sys','subprocess','pathlib','io','socket','urllib','http','ftplib','shutil','pickle','marshal','multiprocessing','ctypes','inspect','builtins','importlib','resource','signal','tempfile','runpy','webbrowser','asyncio'}
    banned_calls={'open','input','print','eval','exec','compile','globals','locals','vars','dir','getattr','setattr','delattr','breakpoint','help'}
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            roots=[a.name.split('.')[0]for a in n.names]if isinstance(n,ast.Import)else[(n.module or '').split('.')[0]]
            if isinstance(n,ast.ImportFrom)and n.level:raise ValueError('Relative imports prohibited')
            if any(m in banned_modules or m not in sys.stdlib_module_names for m in roots):raise ValueError('Unsupported IO/process/reflection/nonstdlib import')
        if isinstance(n,ast.Attribute)and n.attr.startswith('__'):raise ValueError('Reflection prohibited')
        if isinstance(n,ast.Name)and(n.id.startswith('__')or n.id in banned_calls):raise ValueError('IO/reflection prohibited')
    compile(tree,'<candidate>','exec')
    return source


def sandbox_profile():
    root=json.dumps(str(Path(sys.base_prefix).resolve()))
    return '(version 1)(deny default)(allow file-read*)(allow process-exec(subpath '+root+'))(allow sysctl-read)(allow signal(target self))'


def grade_with_contract(text,name,pairs,*,require_new_return_list,preserve_inputs=False,timeout=4,memory_limit_bytes=512*1024*1024):
    if type(require_new_return_list)is not bool:raise ValueError("Explicit boolean fresh-list contract required")
    try:source=extract_source(text,name)
    except SyntaxError as e:return {'correct':False,'status':'compile_error','cases_completed':0,'error':str(e)[:500]}
    except(ValueError,TypeError)as e:return {'correct':False,'status':'source_error','cases_completed':0,'error':str(e)[:500]}
    if not Path('/usr/bin/sandbox-exec').is_file():raise RuntimeError('Required OS execution restriction unavailable;never run generated code directly')
    payload={'source':source,'name':name,'cases':pairs,'preserve_inputs':preserve_inputs,'require_new_return_list':require_new_return_list}
    with tempfile.TemporaryDirectory(prefix='contract_function_')as work:
        command=['/usr/bin/sandbox-exec','-p',sandbox_profile(),sys.executable,'-I','-S','-B','-X','utf8','-c',WORKER]
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,cwd=work,env={'PATH':'/usr/bin:/bin','LANG':'en_US.UTF-8'})
        deadline=time.monotonic()+timeout
        with ThreadPoolExecutor(max_workers=1)as pool:
            future=pool.submit(process.communicate,json.dumps(payload))
            while not future.done():
                try:oversize=psutil.Process(process.pid).memory_info().rss>memory_limit_bytes
                except psutil.NoSuchProcess:oversize=False
                if oversize or time.monotonic()>=deadline:
                    process.kill();future.result()
                    return {'correct':False,'status':'memory_limit'if oversize else'timeout','cases_completed':0}
                time.sleep(0.02)
            stdout,stderr=future.result()

    if process.returncode!=0:return {'correct':False,'status':'timeout'if process.returncode in(-9,-24)else'execution_process_error','cases_completed':0,'returncode':process.returncode,'stderr':stderr[:500]}
    try:value=json.loads(stdout)
    except json.JSONDecodeError:return {'correct':False,'status':'execution_protocol_error','cases_completed':0,'stderr':stderr[:500]}
    assert isinstance(value,dict)and type(value['correct'])is bool and value['status']in {'passed','wrong_answer','input_mutation','runtime_error','import_error','unexpected_IO','returned_input_alias','return_type_error'}
    return value
