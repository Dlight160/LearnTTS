# QA Log

## 2026-05-11

### Q: 为什么你每次操作一个命令都要征求我的同意？
A: 这是 Claude Code 的权限系统。`settings.local.json` 中的 `permissions.allow` 白名单规则默认是**精确匹配**，只有命令字符串完全匹配规则时才自动放行。之前我们的规则太窄（如 `Bash(git -C "D:/Work/Code/LearnTTS" rev-parse --git-dir)` 只匹配那一条 git 命令），换个参数就触发审批。解决方案是改成了宽泛的通配规则（如 `Bash(git *)`、`Bash(ls *)`），覆盖一类操作而非单个命令。

### Q: 和 Claude Code 的聊天记录能放到 git 仓库里同步吗？
A: 不能直接放。聊天记录存储在 `~/.claude/projects/D--Work-Code-LearnTTS/` 和 `~/.claude/history.jsonl`，在用户目录下，不在项目文件夹内。替代方案是利用 CLAUDE.md 已规定的 QA_Log.md 机制——每次概念性/学习性问题解答后自动追加记录，这个文件会被 git 追踪，从而实现知识跨设备同步。
