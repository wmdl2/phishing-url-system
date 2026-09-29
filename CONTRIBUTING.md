# 多人协作开发

仓库保持私有。仓库拥有者可以在 GitHub 的 Settings → Collaborators 中邀请协作者，授予写入权限；无需把仓库改为公开。

## 开发流程

1. 克隆仓库，安装 Python 3.12 和 requirements-dev.txt，激活自己的虚拟环境。
2. 从最新 main 创建任务分支，例如 feat/history-filter 或 fix/url-parser。
3. 先明确任务负责的文件，减少多人同时修改同一模块。
4. 修改后执行 Ruff、Pyright 和 Pytest，提交清楚的中文说明。
5. 推送任务分支并发起 PR，写明变化、验证方法、是否影响特征或模型版本。
6. 由另一位协作者检查后合并；GitHub Actions 会自动运行检查。

~~~bash
git switch main
git pull --ff-only
git switch -c feat/your-task
python -m ruff check .
python -m pyright --pythonpath "$(which python)"
python -m pytest -q
git add 具体文件
git commit -m "描述本次修改"
git push -u origin feat/your-task
~~~

## 模块衔接

- 修改特征名称、顺序或计算方式必须更新特征版本并重新训练；训练和检测必须继续使用同一函数。
- 保留数据来源与哈希；参数只由验证集选，不用最终测试成绩反复选模型。
- 模型文件不在 Git 中传递。协作者下载固定版本数据并本地训练，核对模型元数据。
- 如果需要多人同步训练产物，先确认来源和哈希，再通过双方约定的私有方式传递。

## 提交范围

仅提交代码、测试、公开的数据来源说明、固定依赖和运行说明。提交前查看 git diff --cached --stat 和 git diff --cached。

不要提交 data、artifacts、archive、docs、虚拟环境、检测历史、真实用户网址、账户凭据、课程参考资料、报告或个人学号姓名。不要使用 git add -f 绕过忽略规则。新增忽略范围之外的文件应单独核对。

每个成员的贡献、课程日期和个人学时按真实过程记录，不用提交次数代替个人学时，也不填写未经确认的成员资料。
