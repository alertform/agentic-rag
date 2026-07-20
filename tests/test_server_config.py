from agentic_search import config


def test_server_host_and_port_defaults():
    assert isinstance(config.SERVER_HOST, str) and config.SERVER_HOST
    assert isinstance(config.SERVER_PORT, int) and config.SERVER_PORT > 0
