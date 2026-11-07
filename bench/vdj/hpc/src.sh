# Source this in a job: pins the branch tip and exports SRC, an immutable copy of that commit
# ($WORK/cache/vdjc-src/<sha>), so concurrent jobs never share a checkout that moves under them.
BRANCH=${BRANCH:-claude/vdj-crispr-benchmark-721d3a}
BARE=$WORK/repos/singlet-py.git
for i in 1 2 3; do git -C $BARE fetch -q origin "+refs/heads/$BRANCH:refs/heads/$BRANCH" 2>/dev/null && break; sleep $((RANDOM % 10 + 2)); done
SHA=${SHA:-$(git -C $BARE rev-parse "refs/heads/$BRANCH")}
SRC=$WORK/cache/vdjc-src/$SHA
if [ ! -d $SRC ]; then
  tmp=$SRC.tmp.$$; mkdir -p $tmp
  git -C $BARE archive $SHA python bench tests | tar -x -C $tmp
  mv -T $tmp $SRC 2>/dev/null || rm -rf $tmp
fi
echo "[src] $BRANCH $SHA"
export SRC SHA
