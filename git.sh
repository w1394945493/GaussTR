git checkout -m -b experiment
git add .
git commit -m ''
git push origin experiment


# 断开故障的 VS Code 凭证助手
unset GIT_ASKPASS
unset SSH_ASKPASS

# 提交代码 
git push origin experiment

# 开启凭证存储
git config --global credential.helper store


# todo 确保当前所在分支
git checkout master
# todo 拉取远程最新的 experiment 分支
git fetch origin
# todo 执行合并
git merge origin/experiment

