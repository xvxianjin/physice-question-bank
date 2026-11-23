# M1–M4 执行记录（2026-11-22）

## 目标

在保持数学题库原功能不变的前提下，完成四门课程的网页切换、物理写库隔离、物理导入/OCR/AI 接线、物理组卷与 PDF/Word 导出。

## 交付范围

- M1：课程切换与物理只读浏览。四门课程（数学、国内高中物理、AP C 力学、AP C 电磁）可在页面切换；国内物理显示人教版（2019）六册目录与省份组；AP 显示 2027 CED 单元与 `ap_exam` 模板。
- M2：物理写库全链路按 `course_id` 隔离。创建、更新、列表、详情、统计、查重、备份、分享与 JSON 交换均保留课程归属；跨课程教材映射和知识点路径被拒绝。
- M3：物理导入/OCR/AI。课程化分类与解答提示词，不使用数学知识点目录；工程样本 `tests/fixtures/physics_engineering_sample.json` 共 120 题（数学 30、国内物理 40、AP 力学 25、AP 电磁 25）。
- M4：物理组卷与导出。AP 卷强制 42 道选择题 + 4 道自由作答题，两部分各占 50%，且 Section I 恒在 Section II 之前；国内物理禁止使用 19 题数学模板。

## 实际验证

Python：`.\.工作区\physics-foundation-venv\Scripts\python.exe -X utf8 -B`

| 组合 | 结果 |
|---|---|
| 课程、迁移、API、课程持久化 | 429 passed, 11 skipped |
| 组卷、导出、选题、恢复 | 262 passed, 3 skipped |
| 导入、分类、文档、OCR | 413 passed, 2 skipped |
| 前端可用性 | 47 passed |
| 组卷模板、AP 卷序与自定义题型渲染 | 54 passed, 1 skipped |

其他检查：

- `git diff --check` 通过。
- `node --check static/js/*.js` 全部通过。
- 真实 Edge 浏览器复测四课程切换、国内物理目录、AP 模板与切回数学后的教材恢复；截图见本目录 `m1-*.png`。
- 独立模型 Testing & Scoring 最终 **10/10**；评审前发现的“自定义题型被计入自由作答却未渲染”已修复并加回归。
- 隔离副本端到端烟测（`.工作区/physics-foundation-source`，独立数据库）：schema 16→17 备份优先迁移成功；创建国内物理题返回 200；物理列表可见、数学列表不可见；跨课程详情返回 404；物理与数学统计各自隔离。正式库 `math_question_bank.db` 未被写入。

## 环境限制（未计入代码通过）

- Windows 无符号链接特权：`test_full_backup.py` 等 symlink 用例出现 `WinError 1314/5`，HEAD 基线同样复现。
- PyMuPDF/PIL 渲染类测试与 `test_source_review_images.py`、`test_pdf_page_vision.py`、`test_pdf_detail_flow.py`、`test_launcher_ports_integration.py` 在本机存在与 HEAD 基线一致的失败，未作为本轮回归通过项。
- 未使用真实付费模型评测 M3 的 precision/recall/F1；120 题只验证了样本结构与解析入口。

## 未完成/边界

- 其余 21 个省级行政区尚未录入省份组；当前 4 组覆盖 10 省。
- 省份组的时长、满分、题型题量目前记录在 `source.document` 文本中，尚未拆成结构化 `exam_profile`（M0 深化项）。
- AP FRQ 的逐题分值细目仍需人工按官方 CED 复核。
- 本轮未提交、未推送、未创建 Tag/Release；正式库未写入。
