import json
import matlibplots.conv_plot_utils as cpu

# factest
f0, fe, nfs = .2, .8, 7
datastr = 'results/factest{0}{1}{2}K30'.format(f0, fe, nfs)
fjs = open(datastr)
fctdct = json.load(fjs)
fjs.close()
cpu.para_plot(xlabel='% \\cfac', ylabel='% \\apprxer', title='(a)',
              logscaley=10, tikzfile='errvsfac.tikz', fignum=123, **fctdct)


# nutest
datastr = 'results/nutest0.01258925411790.0007943282347249K203040'
fjs = open(datastr)
fctdct = json.load(fjs)
fjs.close()
cpu.para_plot(xlabel='\\mu', ylabel='\\apprxer', title='(b)',
              logscaley=10, tikzfile='errvsnu.tikz', fignum=124, **fctdct)
