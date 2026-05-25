
$WORKRTREE_NAME = $1

git worktree add worktrees/$WORKRTREE_NAME

cd worktrees/$WORKRTREE_NAME

ln -s ../../.claude .claude
ln -s ../../artifacts artifacts
