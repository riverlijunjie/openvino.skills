---
name: dev_code_review
description: >
  Unified skill for Code review for OpenVINO.
---

## Workflow

1. **Checkout code**: Download code from the relevant branch or PR, you should use git to manage your local copy and ensure you have the latest changes.
      - git remote add <username> <repo_url>
      - git fetch <username>
      - git checkout -b <branch_name> <username>/<branch_name>
2. **Understand PR**: Read the PR description, check linked issues, and review code changes to understand the context and goals of the optimization.
3. **Analyze PR**: analyze the code changes to identify:
     - Code style issues: should aligned with the same styles of the same directory.
     - Functionality issues: If the code changes are related to new features, check if the implementation is correct and complete.
     - Check if it is related to the new models support, if yes, please read materials related the new models feature:
        - Special documentation, such as feature_gemma4_moe_analysis.md
        - If no special documentation, please find original modeling files to analyze the feature requirement: local ~/workspace/transformers/src/transformers/models or https://github.com/huggingface/transformers/blob/main/src/transformers/models
     - If the code changes are related to optimization, check if the optimization is effective and does not introduce new bugs.
     - Potential bugs: check if the code changes could introduce bugs, look for logic errors, edge cases, or incorrect assumptions that could lead to incorrect results or crashes.
     - Potential performance bottlenecks: such as inefficient algorithms, redundant computations, or suboptimal memory access patterns. 

4. **Give code review comments**: Give a list of code review comments based on the analysis, including:
     - Code style issues: point out any code style issues and suggest improvements.
     - Functionality issues: point out any functionality issues and suggest improvements.
     - Potential bugs: point out any potential bugs and suggest improvements.
     - Potential performance bottlenecks: point out any potential performance bottlenecks and suggest improvements.
     - Optimization suggestions: if the code changes are related to optimization, suggest potential optimizations and improvements.
5. **Document**: Update `SUMMARY.md` with results; keep SKILL.md as concise reference

### Rules
- Follow existing code style and conventions
- Don't modify code but only provide comments and suggestions
- Don't create new commits or push code
- Don't modify oneDNN code directly


## Related Docs

- `feature_xxx_analysis.md` — Some special feature analysis material, such as feature_gemma4_moe_analysis.md for MoE optimization analysis
- `SUMMARY.md` — Full work summary with detailed architecture, performance tables, and history

