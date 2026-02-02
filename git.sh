git checkout -m -b experiment
git add .
git commit -m ''
git push origin experiment

# todo 确保当前所在分支
git checkout master
# todo 拉取远程最新的 experiment 分支
git fetch origin
# todo 执行合并
git merge origin/experiment