from anodyne_storage.secrets import FernetSecretStore
from cryptography.fernet import Fernet


def test_encrypt_decrypt_roundtrip() -> None:
    store = FernetSecretStore(Fernet.generate_key())
    ref = store.encrypt("sk-secret")
    assert ref != "sk-secret"
    assert store.decrypt(ref) == "sk-secret"


def test_ciphertext_is_not_plaintext_substring() -> None:
    store = FernetSecretStore(Fernet.generate_key())
    assert "sk-secret" not in store.encrypt("sk-secret")
