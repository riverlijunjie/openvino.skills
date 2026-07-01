---
name: dev_port_master_to_ovmx
description: Port some commit or feature of openvino master branch to openvino.mx new branch, and make sure it works in openvino.mx.
---

When want to port some commits or features from OpenVINO master branch to OpenVINO.mx master branch, the following steps can be taken:

1. **Top priority policies**
   - Don't create new commit, only modify code locally and remote copy to remote machine for build and test.
   - Don't push code to remote repository.
   - Follow the same code style and conventions used in the existing codebase to maintain consistency.

2. **Search all commits related the features**: Ensure that you have identified all relevant commits in the OpenVINO master branch that pertain to the feature or bug fix you want to port. 
    You must point out the openvino directory and search all commits/source code related to the feature or bug fix you want to port. 
    This may involve using git log, git diff, and other git tools to track changes.
        - Use `git log` to find the commit hashes of the relevant changes.
        - Use `git diff` to see the changes made in each commit.
        - Generate patch files for the commits you want to port using `git format-patch` or similar tools.

3. **Check differences between openvino.mx and openvino**: 
    - Generally, the openvino.mx codebase is a fork of the OpenVINO master branch, but there may be differences due to customizations or modifications made in openvino.mx.
    - You need to check the differences between the two codebases to ensure that the changes you are porting will not conflict with existing code in openvino.mx.

4. **Apply patches from openvino to openvino.mx**: 
    - Always ask user whether create new branch and the new branch name before apply the patches.
    - Apply the patches you generated from the OpenVINO master branch to the openvino.mx codebase. This may involve using `git apply` or manually applying the changes.
    - After applying the patches, you may need to resolve any conflicts that arise due to differences between the two codebases. 

5. **Build and test**:
    - Build the openvino.mx codebase after applying the patches to ensure that the changes have been integrated correctly and that there are no build errors.
    - If possible, test the changes locally or remotely to ensure that they work as expected and do not introduce any new issues.

6. **Document optimization**:
    - Summarize the patches applied and their details to "SUMMARY.md"

Keep explanations conversational. For complex concepts, use multiple analogies.


