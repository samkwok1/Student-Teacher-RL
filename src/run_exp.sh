

#!/bin/bash
# chmod +x run_exp.sh
# run_exp.sh

# Define hyperparameters for tuning
# gamma: 0.9
# alpha: 0.1
# epsilon: 0.1
parent_reliabilities=(1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
pre_advice_epsilons=(0.2 0.25 0.3 0.35 0.4 0.45 0.5)
post_advice_weights=(0.05 0.1 0.15 0.2)


# Loop through beta and learning rate values
for parent_reliability in "${parent_reliabilities[@]}"; do
    for pre_advice_epsilon in "${pre_advice_epsilons[@]}"; do
        exp_name="${parent_reliability}_${pre_advice_epsilon}"
        echo "Running experiment $exp_name"

        # Run your Python script with parameters
        python3 main.py \
            --Q_hyper.parent_reliability=$parent_reliability \
            --Child_params.pre_advice_epsilon=$pre_advice_epsilon \
            --exp_name=$exp_name

    done
done