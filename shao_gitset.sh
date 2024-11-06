
git config --global --add safe.directory '*'
git config --global user.name "Jiang Shao"
git config --global user.email jiangs@nvidia.com

chmod -R 777 ../latest_trtllm_github >/dev/null 2>&1
git config --add core.filemode false