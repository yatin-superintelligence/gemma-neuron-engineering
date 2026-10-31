"""Encode the explicit partial assistant source without an answer token."""
import hashlib
import json


def encode(tokenizer, row, device='cpu'):
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    text = prompt + row['program_prefix']
    encoded = tokenizer(text, add_special_tokens=False, return_tensors='pt')
    ids = encoded['input_ids'][0].tolist()
    assert tokenizer.decode(ids, skip_special_tokens=False) == text
    metadata = {'id': row['id'], 'input_tokens': len(ids),
        'template_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'input_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        'program_prefix_sha256': hashlib.sha256(row['program_prefix'].encode()).hexdigest()}
    return encoded.to(device), metadata
