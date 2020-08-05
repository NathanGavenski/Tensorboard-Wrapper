python test.py
ret=$?

if [ $ret -ne 0 ]; then
     echo "It Failed"
else
    # Get current tag number
    VERSION=`git describe --abbrev=0 --tags`
    
    # Split numbers by dot
    VERSION_BITS=(${VERSION//./ })
    
    # Add one to the last version
    VNUM1=${VERSION_BITS[0]}
    VNUM2=${VERSION_BITS[1]}
    VNUM3=${VERSION_BITS[2]}
    VNUM3=$((VNUM3+1))
    NEW_TAG="$VNUM1.$VNUM2.$VNUM3"
    
    # Create tag and deploy
    git tag $NEW_TAG
    git push; git push --tags
fi