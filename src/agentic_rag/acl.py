"""ACL 权限层:语料目录的 acl.json 把 glob 模式映射到 access 级别。

企业刚需:检索结果必须按提问者可见范围过滤,否则就是数据泄露。
acl.json 形如 {"finance/*": "confidential", "*.pdf": "internal"},按声明顺序首个匹配生效。
"""
import fnmatch
import json
from pathlib import Path

DEFAULT_ACCESS = "public"


def load_acl(docs_dir: Path) -> list[tuple[str, str]]:
    """读 docs_dir/acl.json(有序 glob → access 规则);无文件返回空规则。"""
    acl_file = docs_dir / "acl.json"
    if not acl_file.is_file():
        return []
    data = json.loads(acl_file.read_text(encoding="utf-8"))
    return list(data.items())


def access_for(source: str, rules: list[tuple[str, str]]) -> str:
    """首个匹配的 glob 规则生效;无匹配默认 public。"""
    for pattern, access in rules:
        if fnmatch.fnmatch(source, pattern):
            return access
    return DEFAULT_ACCESS
