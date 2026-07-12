import pytest
from anodyne_core.ports import AuthorizationPolicy, LLMProvider, ObjectStore, SecretStore


@pytest.mark.parametrize("cls", [ObjectStore, SecretStore, LLMProvider, AuthorizationPolicy])
def test_ports_are_abstract(cls: type) -> None:
    with pytest.raises(TypeError):
        cls()
