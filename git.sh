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