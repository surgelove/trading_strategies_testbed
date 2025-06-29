#!/bin/bash
# filepath: /Users/code/source/trading_strategies_testbed/Weekly/auto.sh

# SSH to Linode, wait 5 seconds, then run ls
ssh linode << EOF
tmux kill-server
tmux kill-session -t gatherer
cd aia
tar -zcvf "yyyy-mm-dd.tar.gz" db
EOF
scp -rp linode:aia/yyyy-mm-dd.tar.gz /Users/code/Downloads
ssh linode << 'EOF'
cd aia
rm yyyy-mm-dd.tar.gz
cd db
rm *
tmux new-session -d -s gatherer 'cd /home/sl/aia && python3 linode_gatherer.py'
EOF
if [ "$(date +%u)" -eq 5 ]; then
    FRIDAY_DATE=$(date +%F)
else
    FRIDAY_DATE=$(date -v -Fri +%F)
fi
echo "Friday date: $FRIDAY_DATE"
mv /Users/code/Downloads/yyyy-mm-dd.tar.gz /Users/code/Downloads/$FRIDAY_DATE.tar.gz