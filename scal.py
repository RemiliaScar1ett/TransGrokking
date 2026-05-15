import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EVENT_PATH = "events.out.tfevents.1778784604.autodl-container-17784a95a9-7d1593cb.37019.0" 

def smooth_curve(scalars, weight=0.6):
    if not scalars:
        return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    ea = EventAccumulator(EVENT_PATH)
    ea.Reload()

    train_events = ea.Scalars('Acc/Train')
    val_events = ea.Scalars('Acc/Val')
    weight_events = ea.Scalars('Dynamics/WeightNorm')

    train_steps = [e.step for e in train_events]
    train_vals = [e.value for e in train_events]
    val_steps = [e.step for e in val_events]
    val_vals = [e.value for e in val_events]
    weight_steps = [e.step for e in weight_events]
    weight_vals = [e.value for e in weight_events]

    train_smooth = smooth_curve(train_vals)
    val_smooth = smooth_curve(val_vals)
    weight_smooth = smooth_curve(val_vals)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    color_raw = '#f8bbd0'
    color_smooth = '#e91e63'

    axes[0].plot(train_steps, train_vals, color=color_raw, linewidth=1.5)
    axes[0].plot(train_steps, train_smooth, color=color_smooth, linewidth=2)
    axes[0].set_title('Acc/Train', loc='left', pad=10, fontsize=12)
    
    axes[1].plot(val_steps, val_vals, color=color_raw, linewidth=1.5)
    axes[1].plot(val_steps, val_smooth, color=color_smooth, linewidth=2)
    axes[1].set_title('Acc/Val', loc='left', pad=10, fontsize=12)

    axes[2].plot(weight_steps, weight_vals, color=color_smooth, linewidth=2)
    axes[2].set_title('Dynamics/WeightNorm', loc='left', pad=10, fontsize=12)

    for ax in axes:
        ax.grid(True, color='#eeeeee', linewidth=1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#bdbdbd')
        ax.spines['left'].set_color('#bdbdbd')
        ax.tick_params(axis='both', colors='#424242')

    plt.tight_layout()
    plt.savefig('accuracy_curves.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()