#!/bin/bash
# filepath: /Users/code/source/trading_strategies_testbed/Weekly/auto.sh

# On linode, kill tmux and zip the database
ssh linode << EOF
tmux kill-server
tmux kill-session -t gatherer
cd aia
tar -zcvf yyyy-mm-dd.tar.gz db
EOF

# In current local directory, remove old files and db directory
rm -f ../yyyy-mm-dd.tar.gz
rm -f -r ../db

# Download the new file from linode
scp -rp linode:aia/yyyy-mm-dd.tar.gz ../

# On linode, remove the compressed file and the database, and restart the gatherer script
ssh linode << 'EOF'
cd aia
rm yyyy-mm-dd.tar.gz
cd db
rm *
tmux new-session -d -s gatherer 'cd /home/sl/aia && python3 linode_gatherer.py'
EOF

# Unzip the downloaded file to a db directory
cd ..
tar -zxvf yyyy-mm-dd.tar.gz

# Get last Friday's date
if [ "$(date +%u)" -eq 5 ]; then
    FRIDAY_DATE=$(date +%F)
else
    FRIDAY_DATE=$(date -v -Fri +%F)
fi
echo "Friday date: $FRIDAY_DATE"

# Rename the generic file to the last Friday's date
mv yyyy-mm-dd.tar.gz $FRIDAY_DATE.tar.gz

# gdrive files upload $FRIDAY_DATE.tar.gz
