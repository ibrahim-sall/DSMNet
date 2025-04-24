#!/bin/bash
# Script to push local repository to multiple GitHub repositories simultaneously
# Created on: April 24, 2025
#
# How to run this script:
#   1. Make sure the script is executable: chmod +x push_to_all_repos.sh
#   2. Run it from the command line: ./push_to_all_repos.sh
#   3. Or run with full path: /home/asfand/Ahmad/DSMNet/push_to_all_repos.sh
#
# This script will simultaneously push your local repository changes
# to both GitHub repositories configured as remotes.

# Define repository paths
LOCAL_REPO_PATH="/home/asfand/Ahmad/DSMNet"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting sync of DSMNet repository to multiple GitHub repositories...${NC}"

# Change to the repository directory
cd $LOCAL_REPO_PATH || { echo -e "${RED}Failed to change to repository directory!${NC}"; exit 1; }

# Check if there are any changes to commit
if git status --porcelain | grep -q .; then
    echo -e "${YELLOW}There are uncommitted changes in your repository.${NC}"
    read -p "Would you like to commit these changes first? (y/n): " commit_changes
    
    if [[ $commit_changes == "y" || $commit_changes == "Y" ]]; then
        read -p "Enter a commit message: " commit_message
        git add .
        git commit -m "$commit_message"
        echo -e "${GREEN}Changes committed successfully!${NC}"
    else
        echo -e "${YELLOW}Continuing without committing changes...${NC}"
    fi
fi

# Push to the first repository (origin)
echo -e "${YELLOW}Pushing to primary repository (origin)...${NC}"
git push origin main
ORIGIN_STATUS=$?

# Push to the second repository (deepvip)
echo -e "${YELLOW}Pushing to secondary repository (deepvip)...${NC}"
git push deepvip main:ahmad-dsmnet
DEEPVIP_STATUS=$?

# Check if both pushes were successful
if [ $ORIGIN_STATUS -eq 0 ] && [ $DEEPVIP_STATUS -eq 0 ]; then
    echo -e "${GREEN}Successfully pushed to all repositories!${NC}"
else
    echo -e "${RED}There were issues pushing to one or more repositories.${NC}"
    [ $ORIGIN_STATUS -ne 0 ] && echo -e "${RED}Failed to push to origin repository.${NC}"
    [ $DEEPVIP_STATUS -ne 0 ] && echo -e "${RED}Failed to push to deepvip repository.${NC}"
fi

echo -e "${YELLOW}Sync operation completed.${NC}"