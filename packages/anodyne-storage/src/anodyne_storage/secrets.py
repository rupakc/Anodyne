from __future__ import annotations

from anodyne_core.ports import SecretStore
from cryptography.fernet import Fernet


class FernetSecretStore(SecretStore):
    """Dev/on-prem symmetric-key secret store. Prod swaps in a Vault/KMS adapter."""

    def __init__(self, key: bytes) -> None:
        self._fernet = Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ref: str) -> str:
        return self._fernet.decrypt(ref.encode()).decode()
