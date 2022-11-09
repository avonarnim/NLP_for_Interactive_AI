In my implementation, I decided to create two separate models for the targets and actions,
respectively. This was done because I wanted to individual outputs' gradients to be computed
independently. This was in an attempt to create predictions for targets and actions that were
fully independent from one another. Additionally, when first setting up my implementation, I
wanted to leave open the option of having different hyperparameters for actions and targets,
which was easier to specify by just creating two different models.

During my initial implementation, I had used a high number of training epochs (100), which took
a very long time to run (on average 3 minutes per epoch), and I read that switching the
optimiizer from SGD to Adam could improve training efficiency, so I made that switch. That said,
I soon read that Adam automatically tunes parameters independently by output automatically,
reducing the utility of splitting the task into two different models.

My initial testing was done with a learning rate of 0.0001, which yielded a very low accuracy
even over the course of a high number of training epochs. I then changed the learning rate to
0.001, which provided almost immediately satisfactory accuracy on both tasks. After 10 epochs,
I found (train, val) action accuracies of (0.96, 0.93) and (train, val) target accuracies of
(0.73, 0.68). The fact that the accuracy for actions was lower makes sense considering that the
range of possible values for actions was 10x as large as the range of possible values for
targets. The increases in accuracy with increases in number of epochs was not significant for
actions (97, 94) or targets (76, 70). This suggests that instead of more epochs, a more
sophisticated model architecture might be a better way to improve accuracy.

To run: python3 train.py --in_data_fn=lang_to_sem_data.json --model_output_dir=experiments/lstm --batch_size=1000 --num_epochs=3 --val_every=1 --force_cpu
