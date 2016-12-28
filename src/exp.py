
def x(a, b, **kwargs):
    print(a)
    y(10, **kwargs)
def y(r,d=5,e=5,f=5, **kwargs):
    print(d)
    print('success %d %d %d' %(d,e,f))
    z(**kwargs)
def z(d=15, e=51,f=51, **kwargs):
    print('me? %d %d %d' %(d,e,f))

dicdt = {'a':15,'d':5, 'e':10040, 'f':1000, 'g':1}
x(b=11,**dicdt)
