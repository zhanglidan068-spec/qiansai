# Git 与 VS Code 使用方法

项目目录：`D:\codex\kwt_fp32`。远程仓库：<https://github.com/zhanglidan068-spec/qiansai>。

## 在 VS Code 中查看和提交

1. 用 VS Code 打开上述项目文件夹，点击左侧“源代码管理”（`Ctrl+Shift+G`）。
2. 修改代码后，在“更改”中点击文件查看差异，再点击文件旁的 `+` 暂存。
3. 输入提交说明，点击“提交”。提交会先保存在本地。
4. 点击“同步更改”或菜单中的“推送”，将提交上传到 GitHub。初次连接若提示登录，请在 GitHub 页面完成账号授权。

数据集、运行环境、上游权重、`runs/` 和 `hw_model/reports/` 等输出被忽略，不会出现在普通待提交列表中。本地文件不会因此被删除。

## 在 VS Code 终端中操作

```powershell
git status
git remote -v

# 查看修改后再暂存、提交和上传。
git diff
git add .
git diff --cached --stat
git commit -m "Describe the change"
git push
```

首次发布主分支使用 `git push -u origin main`。如果远程已经有提交，先获取并查看远程历史，再合并后推送；不要强制推送覆盖远程内容。

首次提交身份默认采用目标 GitHub 仓库所有者的用户名和 noreply 邮箱，仅配置在本仓库。可以自行调整为 GitHub 账户“Emails”页面提供的身份：

```powershell
git config --local user.name "Your name"
git config --local user.email "Your GitHub commit email"
```

## 换电脑后恢复

```powershell
git clone https://github.com/zhanglidan068-spec/qiansai.git
code .\qiansai
```

克隆只恢复 Git 管理的内容。模型、数据集、环境和历史实验需另行准备，参见主 README。
