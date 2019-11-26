#!/bin/bash
for notebook in `ls examples/*.ipynb`; do
    echo "python3 -m nbconvert --to=notebook --execute examples/$notebook"
    IEX_TOKEN=Tpk_ecc89ddf30a611e9958142010a80043c python3 -m nbconvert --to=notebook --execute $notebook
    # python3 -m nbconvert --to=notebook --execute $notebook > /dev/null 2>&1 || echo "failed: $notebook"
done
