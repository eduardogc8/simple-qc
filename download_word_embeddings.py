import os
muse_embeddings_path= 'muse_embeddings'

def download_if_not_existing(langs=['en', 'de', 'pt']):
    if not os.path.isdir(muse_embeddings_path):
        os.mkdir(muse_embeddings_path)
    for lang in langs:
        if not os.path.isfile('%s/%s' % (muse_embeddings_path, 'wiki.multi.%s.vec' % lang)):
            url = 'https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.%s.vec' % lang
            os.system('wget -N -q -P %s %s' % (muse_embeddings_path, url))


if __name__ == '__main__':
    download_if_not_existing()
