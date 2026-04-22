import os
from cryptography.fernet import Fernet

KEY_FILE = "fernet.key"

def _get_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
    else:
        with open(KEY_FILE, "rb") as f:
            key = f.read()
    return key

_fernet = Fernet(_get_key())

def encrypt_text(text: str) -> str:
    if not text:
        return text
    return _fernet.encrypt(text.encode()).decode()

def decrypt_text(encrypted_text: str) -> str:
    if not encrypted_text:
        return encrypted_text
    return _fernet.decrypt(encrypted_text.encode()).decode()
