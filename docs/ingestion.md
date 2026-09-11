# 文件发现与基础加载

`FileSystemSourceProvider` 接受目录、显式文件和 glob。每个 `SourceDescriptor` 包含 SHA-256 内容摘要、UTC 修改时间、大小、规范化相对路径和稳定 source ID。

```python
from offline_rag.sources import DiscoveryOptions, FileSystemSourceProvider

provider = FileSystemSourceProvider(
    ["./documents"],
    namespace="company-knowledge",
    options=DiscoveryOptions(
        recursive=True,
        allowed_extensions=(".txt", ".md", ".markdown", ".pdf", ".docx"),
        ignore_patterns=("**/.git/**", "**/*.tmp"),
    ),
)
sources = list(provider.discover())
```

默认不读取隐藏路径，不跟随软链接。显式启用软链接后仍会通过 inode 去重阻止目录循环，并拒绝根目录之外的链接目标。扫描期间发生变化的文件会失败，不会把不一致的 hash 和内容写入索引。

`TextLoader` 支持 UTF-8、UTF BOM 和本地编码检测；loader 会重新校验发现阶段的内容摘要。`MarkdownLoader` 与 `MarkdownParser` 保留标题、段落、列表和代码围栏结构。基础 normalizer 统一 NFC Unicode、换行和非法控制字符，代码与表格的空白布局不做压缩。

当前内置格式还包括：文本型 PDF（保留一基页码）、DOCX（按正文顺序保留标题/段落/表格）、HTML 正文结构、CSV 行记录，以及 JSON/JSONL 记录边界。扫描型 PDF 不会静默产出空索引，而是明确提示启用后续 OCR extra。

高级切块提供 Page Aware、Paragraph Packing、Table Row 和 Parent Child。Parent Child 的每个 child 都保存稳定 `parent_id` 与 parent 原文，后续 organizer 可以合并多个 child 命中而不依赖额外数据库。精确内容去重默认启用；近似去重必须显式设置阈值。
