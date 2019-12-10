a = tiff.imread('/Users/stevenban/Downloads/Good Rainbow Worms/20181001-col00-0/20181001-col00-0.tif')
s = a.shape


a = a.reshape(-1, a.shape[-2], a.shape[-1])
a = a.reshape(s)
