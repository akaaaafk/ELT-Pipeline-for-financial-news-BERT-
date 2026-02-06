# 推送到 GitHub 的步骤

本地仓库已初始化并完成首次提交，按下面步骤即可推到 GitHub。

## 1. 在 GitHub 上新建仓库

1. 打开 https://github.com/new  
2. 填写 **Repository name**（例如：`ieor5400-data-engineering-project`）  
3. 选择 **Public**，**不要**勾选 “Add a README”  
4. 点击 **Create repository**

## 2. 在本地添加远程并推送

在项目目录下执行（把 `你的用户名` 和 `仓库名` 换成你的）：

```powershell
cd "d:\Columbia\Fall2025\5400\project"

# 添加远程（替换为你的 GitHub 用户名和仓库名）
git remote add origin https://github.com/你的用户名/仓库名.git

# 推送到 GitHub（主分支）
git push -u origin master
```

若 GitHub 上默认分支是 `main`，可先改成本地分支名再推送：

```powershell
git branch -M main
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main
```

## 已包含在仓库中的内容

- **Submission.ipynb**（主要提交笔记本）
- **README.md**（项目说明）
- **.gitignore**（排除大文件）
- 其他笔记本、Flask 应用、CSV 样本、PDF、图表等

## 未纳入仓库的大文件

- `layer/`（Parquet 数据）
- `models/`（如 FinBERT 等模型缓存）

若以后需要把这些也放到 GitHub，可使用 [Git LFS](https://git-lfs.github.com/) 或单独的数据/模型仓库。
