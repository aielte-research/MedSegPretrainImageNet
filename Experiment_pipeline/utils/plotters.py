from bokeh.plotting import figure, output_file, save, ColumnDataSource
from bokeh import palettes
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper
import numpy as np
import math
from pathlib import Path

from scipy import stats
import colorcet as cc

def filename_from_path(fpath):
    return Path(fpath).stem#+Path(fpath).suffix

def get_colors(nmbr):
    if nmbr>10:
        return [palettes.Turbo256[int(len(palettes.Turbo256)/nmbr*i)] for i in range(nmbr)]
    else:
        return palettes.Category10[10][:nmbr]

def get_baseline_colors(nmbr):
    return [palettes.Greys256[(194 // nmbr) * i] for i in range(nmbr)]

def get_string_val(lst, i):
    try:
        return lst[i]
    except IndexError:
        return ''

def jitter(lst, width):
    return [x + np.random.uniform(low=-width, high=width) for x in lst]

def scatter(Xs, Ys, xlabel="", ylabel="", title="", width=None, height=None, fpath="./scatter.html", line45=True, labels=[], legend_location="bottom_right", neptune_experiment=None,
                    opacity=0, circle_size=4, x_jitter=0, heatmap = False, palette = cc.CET_L18,
                    boundary_funcs = [], *args, **kwargs):
    if width==None or height==None:
        p = figure(sizing_mode='stretch_both',
                    title = title,
                    tools = "pan,box_zoom,wheel_zoom,xwheel_zoom,ywheel_zoom,save,reset"
                    )
    else:
        p = figure(plot_width = width,
                    plot_height = height,
                    title = title,
                    tools = "pan,box_zoom,wheel_zoom,xwheel_zoom,ywheel_zoom,save,reset"
                    )
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel
    
    X_nans=[[x for x,y in zip(X,Y) if math.isnan(y)] for X,Y in zip(Xs,Ys)]
    Y_nans=[[0 for y in Y if math.isnan(y)] for Y in Ys]

    Xs=[[x for x,y in zip(X,Y) if not math.isnan(y)] for X,Y in zip(Xs,Ys)]
    Ys=[[y for y in Y if not math.isnan(y)] for Y in Ys]
        
    colors = get_colors(len(Xs))
    for i,(X,Y) in enumerate(zip(Xs,Ys)):
        p.circle(jitter(X,x_jitter), Y, size=circle_size, line_width=0, color=colors[i], legend_label = get_string_val(labels,i), alpha=1-opacity)
    
    for i,(X,Y) in enumerate(zip(X_nans,Y_nans)):
        p.dash(jitter(X,x_jitter), Y, size=circle_size, line_width=.5, color=colors[i], legend_label = get_string_val(labels,i), alpha=0, line_alpha=1-opacity)

    p.legend.location = legend_location
    
    
    if line45:
        min_x=min([min(X) for X in Xs])
        max_x=max([max(X) for X in Xs])
        src=ColumnDataSource(data=dict(x=[min_x,max_x], y=[min_x,max_x]))
        p.line("x","y", line_color='black', line_width=2,source=src)
    
    p.add_tools(HoverTool(tooltips = [(xlabel, "@x"),
                                        (ylabel, "@y")]))

    if heatmap:
        x, y = Xs[0], Ys[0]
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        xrange = xmax-xmin
        xmin = min(x)-0.25*xrange
        xmax = max(x)+0.25*xrange
        yrange = ymax-ymin
        ymin = min(y)-0.25*yrange
        ymax = max(y)+0.25*yrange

        X, Y = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)

        p.image(image=[np.transpose(Z)], x=xmin, y=ymin, dw=xmax-xmin, dh=ymax-ymin, palette=palette, level="image")

        color_mapper = LinearColorMapper(palette=palette, low=0, high=np.amax(Z))
        color_bar = ColorBar(color_mapper=color_mapper, location = (0,0))

        p.add_layout(color_bar, 'right')

    if len(boundary_funcs)>0:
        x_range=np.linspace(min_x,max_x,100)
        for bf in boundary_funcs:
            src = ColumnDataSource(data=dict(
                x=x_range,
                y=[eval(bf) for x in x_range],
            ))
            p.line('x', 'y', color = 'black', line_alpha = 0.75, line_width=2, source=src, line_dash = 'dashed')

    
    output_file(fpath)
    save(p)
    if neptune_experiment!=None:
        neptune_experiment[filename_from_path(fpath)].upload(fpath)
        #neptune_experiment.sync() 

def general_line_plot(ys, xlabel="", ylabel="", title="", width=None, height=None, labels=[], fpath="", legend_location=None, baselines={}, dashes=["solid"], x=[], neptune_experiment=None, baseline_styles = ['dotted', 'dashed', 'dashdot']):
    if type(ys[0])!=list:
        ys=[ys]
        
    if x==[]:
        x = [list(range(1, len(y) + 1)) for y in ys]
    elif not isinstance(x[0], list):
        x = [x for _ in ys]
    
    if width==None or height==None:
        p = figure(sizing_mode='stretch_both',
                   title = title,
                   tools = "pan,box_zoom,wheel_zoom,xwheel_zoom,ywheel_zoom,save,reset"
                  )
    else:
        p = figure(plot_width = width,
                   plot_height = height,
                   title = title,
                   tools = "pan,box_zoom,wheel_zoom,xwheel_zoom,ywheel_zoom,save,reset"
                  )
    xlabel = xlabel.replace('_', ' ')
    ylabel = ylabel.replace('_', ' ')
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel

    x_len=max([len(y) for y in ys])
    
    colors = get_colors(len(ys))
    dashes=[dashes[i%len(dashes)] for i in range(len(ys))]

    sources = [ColumnDataSource(data=dict(x=x[i],
                                            y=y,
                                            maxim=[max(y) for i in y],
                                            minim=[min(y) for i in y],
                                            argmax=[np.argmax(y)+1 for i in y],
                                            argmin=[np.argmin(y)+1 for i in y],
                                            label=[get_string_val(labels,i).replace('_',' ') for _ in y]
                                            )
                                ) for i,y in enumerate(ys)]
    
    if labels==[] or len(labels)>10:
        for (c, source, dash) in zip(colors, sources, dashes):
            p.line('x', 'y', color=c, line_width=2, source=source, line_dash = dash)
    else:
        for (c, l, source, dash) in zip(colors, labels, sources, dashes):
            p.line('x', 'y', color=c, legend_label=l.replace('_',' '), line_width=2, source=source, line_dash = dash)
        if legend_location is None:
            legend_location = 'top_right' if np.argmax(ys[0]) < np.argmin(ys[0]) else 'bottom_right'
        p.legend.location = legend_location
    
    baseline_colors = get_baseline_colors(len(baselines))
    for i, (name, value) in enumerate(baselines.items()):   
        src = ColumnDataSource(data = {"x": [0,x_len+1],
                                        "y": [value,value],
                                        "maxim": [value,value],
                                        "label": [name.replace('_',' '),name.replace('_',' ')]
                                        })
        p.line("x","y",
                legend_label = name.replace('_',' '),
                line_dash = baseline_styles[i % len(baseline_styles)],
                line_color = baseline_colors[i],
                line_width = 2,
                source = src)
    
    p.add_tools(HoverTool(tooltips = [(xlabel, "@x"),
                                        (ylabel, "@y"),
                                        ("name", "@label"),
                                        ("max", "@maxim"),
                                        ("argmax", "@argmax"),
                                        ("min", "@minim"),
                                        ("argmin", "@argmin")
                                        ],
                            mode='vline'))
    p.add_tools(HoverTool(tooltips = [(xlabel, "@x"),
                                        (ylabel, "@y"),
                                        ("name", "@label"),
                                        ("max", "@maxim"),
                                        ("argmax", "@argmax"),
                                        ("min", "@minim"),
                                        ("argmin", "@argmin")
                                        ]))
    
    output_file(fpath)
    save(p)
    if neptune_experiment!=None:
        neptune_experiment[filename_from_path(fpath)].upload(fpath)
        #neptune_experiment.sync() 