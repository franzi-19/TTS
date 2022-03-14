import os

import umap
import pandas as pd
from bokeh.io import save, export_png
from bokeh.models import (BoxZoomTool, ColorBar, ColumnDataSource, HoverTool,Whisker,
                          OpenURL, ResetTool, TapTool)
from bokeh.palettes import Category10, Category20, Spectral6
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, linear_cmap

CIRCLE_SIZE = 2
CROSS_SIZE = 2
TICK_NUMBER_SIZE = "15pt"

def plot_embeddings(embeds, plot_path, labels=None, filename='plot.png'): # embeds: array([array([coordinate1, co2]), ...])
    model = umap.UMAP()
    projection = model.fit_transform(embeds)

    if labels != None and labels != []:
        source_wav_stems = ColumnDataSource(
                data=dict(
                    x = projection.T[0].tolist(),
                    y = projection.T[1].tolist(),
                    #desc=locations,
                    label=labels
                )
            )
    else:
        source_wav_stems = ColumnDataSource(
                data=dict(
                    x = projection.T[0].tolist(),
                    y = projection.T[1].tolist(),
                    #desc=locations
                )
            )

    # hover = HoverTool(
    #         tooltips=[
    #             #("file", "@desc"),
    #             ("attack", "@label"),
    #         ]
    #     )

    if labels != None and labels != []:
        factors = list(set(labels))
        pal_size = max(len(factors), 3)
        if pal_size <= 10:
            pal = Category10[pal_size]
        elif pal_size <= 20:
            pal = Category20[pal_size]
        else:
            raise ValueError('Too many different labels to plot')

    p = figure(plot_width=1000, plot_height=600)#, tools=[hover,BoxZoomTool(), ResetTool(), TapTool()])

    if labels != None and labels != []:
        if len(list(set(labels))) == 1:
            p.circle('x', 'y',  source=source_wav_stems, size=CIRCLE_SIZE, color='gray')
        else:
            p.circle('x', 'y',  source=source_wav_stems, size=CIRCLE_SIZE, color=factor_cmap('label', palette=pal, factors=factors))
    else:
        p.circle('x', 'y',  source=source_wav_stems)

    p.xaxis.major_label_text_font_size = TICK_NUMBER_SIZE
    p.yaxis.major_label_text_font_size = TICK_NUMBER_SIZE

    file_path = plot_path + filename
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # save(p, file_path, title=title)
    export_png(p, filename=file_path)
    print(f'saved plot at {file_path}')

def plot_box_plot(labels, values, plot_path, filename):
    labels = [label.split('_')[-1] for label in labels]
    df = pd.DataFrame(dict(score=values, group=labels))
    cats = sorted(list(set(labels)))

    # find the quartiles and IQR for each category
    groups = df.groupby('group')
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr

    # find the outliers for each category
    def outliers(group):
        cat = group.name
        return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']
    out = groups.apply(outliers).dropna()

    # prepare outlier data for plotting, we need coordinates for every outlier.
    if not out.empty:
        outx = list(out.index.get_level_values(0))
        outy = list(out.values)

    p = figure(plot_width=1000, plot_height=600, x_range=cats)

    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper.score = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'score']),upper.score)]
    lower.score = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'score']),lower.score)]

    # stems
    p.segment(cats, upper.score, cats, q3.score, color="white", line_color="black")
    p.segment(cats, lower.score, cats, q1.score, color="white", line_color="black")

    # boxes
    p.vbar(cats, 0.7, q2.score, q3.score, fill_color="white", line_color="black")
    p.vbar(cats, 0.7, q1.score, q2.score, fill_color="white", line_color="black")

    # whiskers (almost-0 height rects simpler than segments)
    p.rect(cats, lower.score, 0.2, 0.01, color="white", line_color="black")
    p.rect(cats, upper.score, 0.2, 0.01, fill_color="white", line_color="black")

    # outliers
    if not out.empty:
        p.circle(outx, outy, size=6, color="white", line_color="black")

    p.xaxis.major_label_text_font_size = TICK_NUMBER_SIZE
    p.yaxis.major_label_text_font_size = TICK_NUMBER_SIZE

    file_path = plot_path + filename
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    export_png(p, filename=file_path)
    print(f'saved plot at {file_path}')


# def plot_bar_plot(labels, values, plot_path, filename):
#     # p = figure(x_range=list(set(classes)), plot_width=1000, plot_height=600)
#     # p.vbar(x=classes, top=values)

#     labels = [label.split('_')[-1] for label in labels]#, key=lambda label: int(label[1:3]) if len(label) == 3 else 0)
#     groups = sorted(list(set(labels)))
#     counts = [0] * len(groups)
#     error = [0] * len(groups)

#     import numpy as np
#     for i, unique_label in enumerate(groups):
#         idx = np.where(np.array(labels) == unique_label)[0]
#         std = np.nanstd(np.array(values)[idx])
#         mean = np.nanmean(np.array(values)[idx])
#         error[i] = std
#         counts[i] = mean

#     upper = [x+e for x,e in zip(counts, error) ]
#     lower = [x-e for x,e in zip(counts, error) ]

#     y_range=(min(lower), max(upper))

#     source = ColumnDataSource(data=dict(groups=groups, counts=counts, upper=upper, lower=lower))
#     p = figure(x_range=groups, y_range=y_range, plot_width=1000, plot_height=600)
#     p.vbar(x='groups', top='counts', source=source)
#     p.add_layout(Whisker(source=source, base="groups", upper="upper", lower="lower", level="overlay"))

#     p.xaxis.major_label_text_font_size = TICK_NUMBER_SIZE
#     p.yaxis.major_label_text_font_size = TICK_NUMBER_SIZE

#     file_path = plot_path + filename
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     export_png(p, filename=file_path)
#     print(f'saved plot at {file_path}')

# https://docs.bokeh.org/en/latest/docs/user_guide/styling.html#using-mappers
def plot_embeddings_continuous(embeds, plot_path, labels, title=None):
    model = umap.UMAP()
    projection = model.fit_transform(embeds)

    if isinstance(labels[0], str):
        factors = list(set(labels))
        pal_size = max(len(factors), 3)
        pal = Category10[pal_size]
        mapper = factor_cmap('label', palette=pal, factors=factors)
    else:
        mapper = linear_cmap(field_name='label', palette=Spectral6 ,low=min(labels) ,high=max(labels))
    source = ColumnDataSource(dict(x=projection.T[0].tolist(), y=projection.T[1].tolist(), label=labels))
    hover = HoverTool(tooltips=[("attack", "@label")])

    p = figure(plot_width=1000, plot_height=600, tools=[hover,BoxZoomTool(), ResetTool(), TapTool()])
    p.circle(x='x', y='y', line_color=mapper, color=mapper, source=source)
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8)
    p.add_layout(color_bar, 'right')

    if title: filename = f"plot_{title}.html"
    else: filename = f"{filename}.html"

    file_path = plot_path + filename
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    save(p, file_path, title=title)
    print(f'saved plot at {file_path}')


def plot_two_sets(embeds_1, labels_1, embeds_2, labels_2, plot_path, filename='plot.png'):
    model = umap.UMAP()
    len_1 = len(embeds_1)
    projection = model.fit_transform(embeds_1 + embeds_2)

    data_1 = ColumnDataSource(data=dict(x = projection.T[0].tolist()[:len_1], y = projection.T[1].tolist()[:len_1], label=labels_1))    # , desc=locations_1, label=labels_1))
    data_2 = ColumnDataSource(data=dict(x = projection.T[0].tolist()[len_1:], y = projection.T[1].tolist()[len_1:], label=labels_2))    # , desc=locations_2, label=labels_2))

    factors_1 = list(set(labels_1))
    pal_size_1 = max(len(factors_1), 3)
    pal_1 = Category20[pal_size_1]

    p_1 = figure(plot_width=1000, plot_height=600) #, tools=[hover,BoxZoomTool(), ResetTool(), TapTool()])
    p_1.circle('x', 'y',  source=data_1, color=factor_cmap('label', palette=pal_1, factors=factors_1), size=CIRCLE_SIZE)#, legend_group='label')
    p_1.cross('x', 'y',  source=data_2, color=factor_cmap('label', palette=pal_1, factors=factors_1), size=CROSS_SIZE)#, legend_group='label')
    p_1.xaxis.major_label_text_font_size = TICK_NUMBER_SIZE
    p_1.yaxis.major_label_text_font_size = TICK_NUMBER_SIZE

    p_2 = figure(plot_width=1000, plot_height=600) #, tools=[hover,BoxZoomTool(), ResetTool(), TapTool()])
    p_2.cross('x', 'y',  source=data_2, color=factor_cmap('label', palette=pal_1, factors=factors_1), size=CROSS_SIZE)#, legend_group='label')
    p_2.circle('x', 'y',  source=data_1, color=factor_cmap('label', palette=pal_1, factors=factors_1), size=CIRCLE_SIZE)#, legend_group='label')
    p_2.xaxis.major_label_text_font_size = TICK_NUMBER_SIZE
    p_2.yaxis.major_label_text_font_size = TICK_NUMBER_SIZE

    file_path_1 = plot_path + '1_' + filename
    file_path_2 = plot_path + '2_' + filename
    os.makedirs(os.path.dirname(file_path_1), exist_ok=True)
    os.makedirs(os.path.dirname(file_path_2), exist_ok=True)
    export_png(p_1, filename=file_path_1)
    export_png(p_2, filename=file_path_2)
    print(f'saved plot at {file_path_1} and {file_path_2}')

def just_plot(x, y, plot_path, labels, title=None):
    hover = HoverTool(tooltips=[ ("attack", "@label"),])
    source_wav_stems = ColumnDataSource(data=dict(x=x, y=y, label=labels))

    factors = list(set(labels))
    pal_size = max(len(factors), 3)
    pal = Category20[pal_size]

    p = figure(plot_width=1000, plot_height=600, tools=[hover,BoxZoomTool(), ResetTool(), TapTool()])
    
    p.circle('x', 'y',  source=source_wav_stems, color=factor_cmap('label', palette=pal, factors=factors), legend_group='label')
    p.legend.location = "bottom_right"
    p.legend.click_policy= "hide"

    if title: filename = f"plot_{title}.html"
    else: filename = "plot.html"

    file_path = plot_path + filename
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    save(p, file_path, title=title)
    print(f'saved plot at {file_path}')

def plot_centroids(points, point_labels, centroids, centroid_labels, centroid_colors, plot_path, title):
    hover = HoverTool(tooltips=[ ("attack", "@label"),])

    data_points = ColumnDataSource(data=dict(x=[point[0] for point in points], y=[point[1] for point in points], label=point_labels))
    data_centroids = ColumnDataSource(data=dict(x=[cent[0] for cent in centroids], y=[cent[1] for cent in centroids], label=[f"{centroid_colors[idx]} ({label[0]},{label[1]})" for idx, label in enumerate(centroid_labels)], color=centroid_colors))

    factors = list(set(point_labels))
    pal_size = max(len(factors), 3)
    pal = Category20[pal_size]

    p = figure(plot_width=1000, plot_height=600, tools=[hover,BoxZoomTool(), ResetTool(), TapTool()])
    
    p.circle('x', 'y',  source=data_points, color=factor_cmap('label', palette=pal, factors=factors), legend_group='label')
    p.cross('x', 'y',  source=data_centroids, color=factor_cmap('color', palette=pal, factors=factors), size=15)#,legend_group='color')
    p.legend.location = "bottom_right"
    p.legend.click_policy= "hide"

    if title: filename = f"plot_{title}.html"
    else: filename = "plot.html"

    file_path = plot_path + filename
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    save(p, file_path, title=title)
    print(f'saved plot at {file_path}')
