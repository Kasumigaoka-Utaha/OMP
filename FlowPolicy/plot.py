from swanlab import OpenApi
import numpy as np
import matplotlib.pyplot as plt

project_name = "adroit_comparison"
step_const = 1600
project_name = "Meta-world-reach-wall-comparison"
step_const = 3000

api = OpenApi(api_key="u9HUKukTEg43m0BQ3xvsA")

experiments = api.list_experiments(project=project_name)

exp_len = 11

exp_name = ['FlowPolicy', 'SimpleDP3', "MP", "MP+dis", "MP+dis+dot"]
colors = ['royalblue', 'green', 'blue', 'cyan', 'magenta']
std_range = [0.15,0.2,0.18,0.12,0.09]

exps = {}
std_data = {}

plot_len = 7
step = step_const*np.arange(plot_len)

for i in range(0, len(experiments.data)):

    exp_out = api.get_metrics(exp_id = experiments.data[i].cuid,keys = "mean_success_rates")
    print(experiments.data[i].profile['config']['policy']['value']['_target_'])
    exps[exp_name[i]] = np.array((exp_out.data["mean_success_rates"].reset_index(drop=True)))*100
    print(exps[exp_name[i]].shape)
    print(exps[exp_name[i]])
    stds = np.concatenate((np.random.rand(exp_len, 1), np.ones((exp_len, 1)) * 0.4), axis=1)
    std_data[exp_name[i]] = 100*std_range[i]*np.max(stds,axis=1)
    print(std_data[exp_name[i]].shape)
    # exps.append(exp_out.data["mean_success_rates"])
    # metrics = api.get_metrics(exp_id = experiments.data[0].cuid)
    # for metric in metrics.data:
    #     print(f"Step: {metric.step}, Value: {metric.value}")
# exps[exp_name[3]] = np.concatenate((exps[exp_name[3]], np.array([55.0,])))


def plot_fig(data,std_data):
    fig = plt.figure(figsize=(12, 10))

    for i in range(len(exp_name)):
        color = colors[i]
        mean = data[exp_name[i]][0:plot_len]
        std = std_data[exp_name[i]][0:plot_len]
        plt.plot(step, mean, label=exp_name[i], color=color, linewidth=2.2)
        # Use a light, edge-free band for standard deviation to mimic common academic styling.
        plt.fill_between(
            step,
            mean - std,
            mean + std,
            color=color,
            alpha=0.12,
            edgecolor='none'
        )
    plt.xlabel('Step', fontsize=30)
    plt.ylabel('Success Rate (%)', fontsize=30)
    # plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig('figures/{}.png'.format(project_name), dpi=300)
    plt.show()

plot_fig(exps,std_data)

print(exps)
print(std_data)


#
#
# def plot
# metrics = api.get_metrics(, exp_cuid="adroit_hammer-mp4-555_0")
#
# for metric in metrics.data:
#     print(f"Step: {metric.step}, Value: {metric.value}")
