from adaptive_agent.skills import SkillStore
from adaptive_agent.schemas import ToolSpec


def spec() -> ToolSpec:
    return ToolSpec(name="adder", description="adds", code="def run(input):\n    return {}",
                    inputSchema={"type": "object"})


def test_persist_and_reload(tmp_path):
    store = SkillStore(skills_dir=tmp_path)
    store.persist(spec())
    digests = store.load_digests()
    assert digests[0].name == "adder"
    assert digests[0].origin == "generated"


def test_load_spec_body_lazily(tmp_path):
    store = SkillStore(skills_dir=tmp_path)
    store.persist(spec())
    loaded = store.load_spec("adder")
    assert "def run" in loaded.code
    assert loaded.description == "adds"


def test_persist_increments_version(tmp_path):
    store = SkillStore(skills_dir=tmp_path)
    store.persist(spec())
    store.persist(spec())
    assert store.read_manifest("adder").version == 2


def test_persist_overwrites_code(tmp_path):
    store = SkillStore(skills_dir=tmp_path)
    store.persist(ToolSpec(name="adder", description="adds",
                           code="def run(input):\n    return {'v': 1}",
                           inputSchema={"type": "object"}))
    store.persist(ToolSpec(name="adder", description="adds",
                           code="def run(input):\n    return {'v': 2}",
                           inputSchema={"type": "object"}))
    assert "'v': 2" in store.load_spec("adder").code
