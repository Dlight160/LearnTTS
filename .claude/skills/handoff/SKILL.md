---
name: handoff
description: Save current work state to HANDOFF.md — records what's done, in progress, and next steps. Invoke when pausing work or wrapping up a session.
disable-model-invocation: true
---

# /handoff

扫描当前对话和工作状态，写入 `HANDOFF.md` 到项目根目录。

## 格式

```markdown
# Handoff — %Y-%m-%d %H:%M

## 已完成
- 已实现的功能、已修复的问题、已做出的决策

## 进行中
- 正在做的、代码已改但未完成的

## 待做
- 下一步计划、已知问题、未完成的思路

## 关键上下文
- 涉及的文件路径
- 重要的架构决策或注意事项
```

## 要求

- 覆盖当前对话中所有相关工作，不要遗漏
- 描述可操作，不要模糊（"修复了X的bug" 而非 "修复了一些问题"）
- 如果项目根目录下已有 HANDOFF.md，读取当前内容并合并（保留未完成的项目，更新已完成的状态）
- 写入后始终提交到 git
