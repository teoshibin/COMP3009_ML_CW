# Cheat Sheet
Clone repository for the first time
```bash
    git clone {link}
```
Before you start coding everytime
```bash
    git pull
```
Done Coding
```bash
    git add .
    git commit -m "message"
    git push
```
Check status
```bash
    git status
```
Usual scenario of pushing new commits & facing conflict
```bash
    ...
    git push
    # fail to push as the branch is ahead of you
    git pull    # pull to update your head
    # if it succeeds, then continue pushing
    # if it fails, a merge conflict has occured

    # to abort
    git merge --abort

    # or to merge differences
    # - edit files that cause conflicts -
    git add .
    git commit -m "resolve merge conflict in ..."
    git push
```