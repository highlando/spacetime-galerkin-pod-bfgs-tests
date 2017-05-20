from run_burger_optcont import testit
import matplotlib.pyplot as plt

basenu = 5e-3
testitdict = \
    dict(Nq=220,  # dimension of the spatial discretization
         Nts=250,  # number of time sampling points
         # space time domain and plot paras
         dmndct=dict(tE=1., t0=0., x0=0., xE=1.,
                     vertflip=True,
                     vmin=-.1, vmax=1.1,
                     cmapname='viridis'),
         adjplotdict=dict(tE=1., t0=0., x0=0., xE=1.,
                          vertflip=True,
                          vmin=-.5, vmax=.5,
                          cmapname='plasma'),
         inivtype='step',  # 'ramp', 'smooth'
         Ns=120,  # Number of measurement functions=Num of snapshots
         hq=10,  # number of space modes
         hs=10,  # number of time modes
         spacebasscheme='VandL',  # 'onlyV',  # 'onlyL' 'VandL' 'combined'
         plotplease=True, tikzplease=True,
         nu=5e-3,
         alpha=1e-3,
         genpodadj=True, genpodstate=True, genpodcl=True)

tikzprefikz = 'plots/hq{0}hs{0}scheme{2}'.\
    format(testitdict['hq'], testitdict['hs'], testitdict['spacebasscheme'])

testitdict.update(tikzprefikz=tikzprefikz)

testit(**testitdict)

plt.close('all')
testitdict.update(dict(spacebasscheme='onlyV'))

tikzprefikz = 'plots/hq{0}hs{0}scheme{2}'.\
    format(testitdict['hq'], testitdict['hs'], testitdict['spacebasscheme'])

testitdict.update(tikzprefikz=tikzprefikz)

testit(**testitdict)
plt.close('all')


testitdict.update(dict(spacebasscheme='combined'))

tikzprefikz = 'plots/hq{0}hs{0}scheme{2}'.\
    format(testitdict['hq'], testitdict['hs'], testitdict['spacebasscheme'])

testitdict.update(tikzprefikz=tikzprefikz)

testit(**testitdict)
plt.close('all')
