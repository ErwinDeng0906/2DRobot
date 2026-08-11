# 如何把代码提交到 GitHub（给完全新手）

本文教你：**改完本项目代码后，怎样安全地交到 GitHub 上**，让项目负责人（Winston）能审查并合并。

你不需要先搞懂 Git 的全部原理。按步骤做命令即可。  
如果某一步报错，先看文末「常见报错」。

仓库地址（本项目）：

https://github.com/WinstonChen-Positer/RobotArm_SCARA_Control

---

## 0. 先搞清几个词（用人话）

| 词 | 人话解释 |
|----|----------|
| **Git** | 装在你电脑上的「版本记录本」，能记住你改过哪些文件。 |
| **GitHub** | 网上的代码仓库网站，大家把代码放在这里共享。 |
| **仓库 / repo** | 这个项目的全部文件集合。 |
| **克隆 / clone** | 把 GitHub 上的项目完整下载到你电脑。 |
| **分支 / branch** | 在主线旁边开的一条「草稿线」。你在草稿上改，不会立刻弄坏大家用的主线。 |
| **main** | 主线分支。默认认为这里是「能跑、比较稳」的版本。**不要直接往 main 上乱推。** |
| **提交 / commit** | 在本地给改动拍一张「存档快照」，并写一句说明。 |
| **推送 / push** | 把你本地的存档上传到 GitHub。 |
| **Pull Request（PR）** | 在网页上申请：「请把我这条分支合并进 main」。负责人同意后才会进主线。 |

记住一句话：

> **本地改 → 本地 commit → push 到自己的分支 → 开 PR → 等合并。**  
> 不要直接改完就往 `main` 推。

---

## 1. 开始之前：你电脑上要准备好这些

### 1.1 GitHub 账号

1. 打开 https://github.com 注册/登录。
2. 告诉项目负责人你的 GitHub 用户名，请他把你加成仓库协作者（Collaborator），否则你可能推不上去。

### 1.2 安装 Git（Windows）

1. 打开 https://git-scm.com/download/win 下载并安装。
2. 安装时大部分选项保持默认即可。
3. 装好后打开 **PowerShell** 或 **命令提示符**，输入：

```bash
git --version
```

能显示出版本号（例如 `git version 2.x.x`）就说明装好了。

### 1.3 第一次用 Git：写上你的名字和邮箱

只需做一次（把内容换成你自己的）：

```bash
git config --global user.name "你的名字或拼音"
git config --global user.email "你的邮箱@example.com"
```

邮箱建议和 GitHub 账号邮箱一致。

### 1.4 把项目拿到自己电脑（如果还没有）

在你想放项目的文件夹里打开终端，执行：

```bash
git clone https://github.com/WinstonChen-Positer/RobotArm_SCARA_Control.git
cd RobotArm_SCARA_Control
```

如果项目已经在电脑上了，进入项目根目录即可（能看到 `main.py` 的那个文件夹）。

### 1.5 本项目特别注意：不要提交本机配置

每个人电脑上的新松软件路径不一样。正确做法是：

1. 复制 `local_config.example.toml` 为 `local_config.toml`
2. 只在你自己电脑上改 `local_config.toml`
3. **不要**把 `local_config.toml` 提交到 GitHub（已在 `.gitignore` 里忽略）

可以提交的是模板：`local_config.example.toml`。

也不要提交：

- 密码、密钥、许可证文件
- 很大的无关文件、个人实验数据
- 自己乱改过的仅本机可用的绝对路径（应写在 `local_config.toml`）

---

## 2. 正式提交流程（每次改代码都走这套）

下面假设你已经在项目根目录（有 `main.py` 的地方）。

---

### 步骤 A：先更新 main，再开自己的分支

目的：基于最新主线开工，减少冲突。

```bash
git checkout main
git pull origin main
```

然后新建并切换到你的分支（名字自己起，建议英文/拼音，用短横线连接）：

```bash
git checkout -b feature/简短描述你的改动
```

例子：

```bash
git checkout -b feature/fix-do-zero-message
git checkout -b fix/connect-button-crash
git checkout -b docs/update-commit-guide
```

命名建议：

- 新功能：`feature/...`
- 修 bug：`fix/...`
- 只改文档：`docs/...`

查看当前在哪个分支：

```bash
git branch
```

前面有 `*` 的就是当前分支。确认不是误停在别人的分支上。

---

### 步骤 B：改代码，并在本地先跑通

1. 在 Cursor / VS Code 里改文件。
2. 尽量自己先跑一下：

```bash
python main.py
```

3. 确认你的改动真的解决了问题，再提交。

---

### 步骤 C：查看改了什么（提交前必看）

```bash
git status
```

会列出：

- 改过的文件（modified）
- 新文件（untracked）

再看具体差别（可选，但推荐）：

```bash
git diff
```

检查清单：

- [ ] 有没有不小心改到无关文件？
- [ ] 有没有出现 `local_config.toml`？（如果出现在待提交列表里，**不要 add 它**）
- [ ] 有没有把自己电脑的 `D:\...` 盘符写进会提交的源码里？（路径应放进 `local_config.toml`）

---

### 步骤 D：把文件放入「暂存区」，再 commit

**只添加你确实要提交的文件**（更安全）：

```bash
git add 路径\到\文件1.py
git add 路径\到\文件2.md
```

如果你很确定当前所有改动都该提交，也可以：

```bash
git add .
```

但用 `git add .` 之前务必再看一眼 `git status`。

然后写提交说明并保存快照：

```bash
git commit -m "用一句话说明为什么改（不要只写改了啥文件名）"
```

好的例子：

```bash
git commit -m "fix: 连接失败时提示检查 local_config.toml"
git commit -m "docs: 补充小白向的 GitHub 提交说明"
git commit -m "feat: DO 清零与使能共用互斥锁，避免抢连接"
```

不好的例子：

```bash
git commit -m "update"
git commit -m "改了一点"
git commit -m "asdf"
```

如果执行 `git commit` 时提示 `nothing to commit`，说明没有成功 `git add`，回到 `git status` 检查。

---

### 步骤 E：推到 GitHub（上传你的分支）

第一次推这个分支：

```bash
git push -u origin 你的分支名
```

例子：

```bash
git push -u origin feature/fix-do-zero-message
```

以后在同一分支上继续改、再 commit 后，可以只打：

```bash
git push
```

推送成功后，浏览器打开仓库页面，通常会看到黄条提示你的分支，可以去开 PR。

---

### 步骤 F：在 GitHub 网页上开 Pull Request（合并申请）

1. 打开：  
   https://github.com/WinstonChen-Positer/RobotArm_SCARA_Control
2. 若出现 **Compare & pull request** 按钮，点它。  
   没有的话：点 **Pull requests** → **New pull request**，  
   base 选 `main`，compare 选你的分支。
3. 填写：
   - **Title**：一句话标题（和 commit 类似即可）
   - **说明**：你改了什么、为什么改、怎么测的（例如「已在本机 python main.py 点连接通过」）
4. 点击 **Create pull request**。
5. 等负责人审查。通过后会 **Merge** 进 `main`。  
   **你自己不要强行合并 main**（除非负责人明确让你做）。

---

### 步骤 G：合并之后，清理并同步本地

在 GitHub 上 PR 合并后，回到本地：

```bash
git checkout main
git pull origin main
```

可选：删除已合并的本地分支

```bash
git branch -d feature/你的分支名
```

云端分支可在 GitHub 的 PR 页面点 **Delete branch**。

之后若要做下一件事，重新从步骤 A 开始（再开新分支）。

---

## 3. 推荐的日常节奏（抄这条就行）

```bash
# 1) 更新主线
git checkout main
git pull origin main

# 2) 开新分支
git checkout -b feature/我的改动名字

# 3) ……这里改代码、测试……

# 4) 查看并提交
git status
git add .
git commit -m "简短说明这次为什么改"

# 5) 上传并开 PR
git push -u origin feature/我的改动名字
# 然后去 GitHub 网页点 Compare & pull request
```

---

## 4. 常见报错（按提示处理）

### 4.1 `Permission denied` / `Authentication failed`（推送被拒绝）

原因：没登录，或没有仓库写权限。

处理：

1. 确认负责人已把你加成协作者。
2. Windows 上可用 GitHub 登录（推荐用 GitHub CLI，或按 GitHub 文档配置 HTTPS 凭据 / SSH）。
3. 推送时按提示登录 GitHub。

### 4.2 `failed to push some refs` / 需要先 pull

原因：云端分支比你新，或你基于过期的 main。

处理（在你自己的功能分支上）：

```bash
git pull origin main
```

若出现冲突，见下一节。解决后再：

```bash
git push
```

### 4.3 合并冲突（conflict）

含义：你和别人改了同一处，Git 不知道留谁的。

处理思路：

1. `git status` 看哪些文件冲突。
2. 打开文件，寻找类似标记：

```text
<<<<<<< HEAD
你的内容
=======
别人的内容
>>>>>>> main
```

3. 手工改成最终正确内容，删掉这些标记行。
4. 然后：

```bash
git add 冲突文件路径
git commit -m "resolve: 解决与 main 的合并冲突"
git push
```

搞不定就找负责人，把冲突文件发过去一起看。

### 4.4 `Please tell me who you are`

还没设置姓名邮箱，回到本文 1.3 节执行两条 `git config`。

### 4.5 不小心 `git add` 了 `local_config.toml`

如果还没 commit：

```bash
git restore --staged local_config.toml
```

如果已经 commit 但还没 push：告诉负责人，不要继续推含密钥/本机路径的提交；必要时一起用正确方式撤掉。

### 4.6 我直接改到 main 上了怎么办？

如果还没 push：

```bash
git checkout -b feature/补救分支名
git push -u origin feature/补救分支名
```

然后把本地 main 恢复成和云端一致（不会的话问负责人，**不要随意 `reset --hard`**）。

如果已经 push 到 main：立刻联系负责人，不要再继续往 main 推。

---

## 5. 本项目提交时的额外规矩

1. **不要直接推 `main`**（除非负责人明确授权的紧急热修）。
2. **不要提交 `local_config.toml`**；只维护/更新 `local_config.example.toml`（如果改了模板）。
3. 换电脑路径相关改动：优先改配置模板与文档，不要把 `D:\某人电脑\...` 写死进会共享的源码。
4. 一次 PR 尽量只做一件事（更好审查）。大改拆成多个小 PR。
5. commit / PR 说明写清楚「为什么」，方便以后查历史。

路径与换机说明见：

- `local_config.example.toml`
- `路径硬编码清单.md`
- `换机路径说明.md`

---

## 6. 最小检查清单（点 Create pull request 前）

- [ ] 当前在自己的功能分支，不是误在 `main` 上推送
- [ ] `python main.py`（或你改动相关的测试）已在本机验证
- [ ] `git status` 干净或只含你打算提交的文件
- [ ] 没有 `local_config.toml`、密钥、许可证
- [ ] commit 说明能看懂
- [ ] 已 `git push` 成功
- [ ] 已在 GitHub 建好 PR，base 是 `main`

---

## 7. 还不会用命令行？

也可以用 Cursor / VS Code 左侧的 **Source Control（源代码管理）**：

1. 确认左下角分支名是你的功能分支（不是 `main`）。
2. 在更改列表里勾选要提交的文件。
3. 输入 commit 说明，点提交。
4. 点 **Sync / Push** 推送。
5. 仍要去 GitHub 网页开 Pull Request（这一步网页上最清楚）。

命令行和图形界面任选一种即可，规则相同：**分支 → commit → push → PR**。
