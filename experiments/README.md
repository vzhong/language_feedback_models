# README

## Behaviour Cloning Baseline

We will use `alfworld` as an example. The other two environments are respectively called `sparse_scienceworld` and `descriptive_touchdown`.

1. Generate gold data for behaviour cloning (BC)

Roll out the trajectories
```
python build_gold_data.py env.name=alfworld
# python build_gold_data.py env.name=descriptive_touchdown
# python build_gold_data.py env.name=sparse_scienceworld env.invalid_action=force
```

Make the behaviour cloning splits
```
python build_bc_data.py --env alfworld
```

2. Train BC model

```
# train model
python train_bc.py env=alfworld train_data=train
# roll out trained model on training envs, alternatively you can set eval.split=dev to roll out on dev envs
python evaluate_bc.py env=alfworld train_data=train eval.split=train
```

We can evaluate this model via:

```
python evaluate_bc.py env=alfworld train_data=train eval.split=dev
python show_eval.py --root outputs/bc/alfworld/ctx=20,data=train/flan_t5/eval/dev/aggregation=mean
```

## Train language feedback model

1. Getting batched LLM feedback

First, get feedback from GPT4 using the OpenAI API.
To do this, create a `conf/secrets.json` as follows:

```
{
  "OPENAI_KEY": "<YOUR_KEY>"
}
```

Next, we will query for LLM feedback:
```
python get_llm_feedback.py outputs/bc/alfworld/ctx=20,data=train/flan_t5/eval/train/aggregation=mean --feedback_style intent --dry_run false
```

Then convert this batched feedback into step-wise dataset:
```
python build_reward_data.py outputs/bc/alfworld/ctx=20,data=train/flan_t5/eval/train/aggregation=mean/feedback/intent data/feedback/alfworld/ctx=20,data=train --p_pos=0.5
```

2. Train LFM

Finally we will train a feedback model:
```
python train_lfm.py env=alfworld p_pos=0.5
```

## Policy Improvement

1. Rollout base policy

First, we will take the BC polciy as the base policy.
Given some rollouts (e.g. on train envs), we will use our learned LFM to extract desirable behaviour:

```
python train_lfm.py env=alfworld p_pos=0.5 "infer_feedback='$PWD/outputs/bc/alfworld/ctx=20,data=train/flan_t5/eval/train/aggregation=mean/'" p_pos=0.0 eval_name=r1
```

2. Create fine-tuning data using LFM

Next, we'll align the extracted feedback with the actual trajectories to create a fine-tuning dataset in `data/bc/alfworld/context=20/r1-gen-0.5-pred_reward.json.bz2`:
```
python build_generated_feedback_finetune_data.py --inp_dir=outputs/reward/alfworld/intent/default_0.5/inferred_feedback/r1 --env=alfworld --strategy pred_reward --p_pos 0.5 --expname r1-
```

3. Train policy for improvement

We can then obtain an improved policy by joining on both this new data and the BC demonstrations:

```
# train model
python train_bc.py env=alfworld train_data=train:r1-gen-0.5-pred_reward
```

We can evaluate this model via:

```
python evaluate_bc.py env=alfworld train_data=train:r1-gen-0.5-pred_reward eval.split=dev
python show_eval.py --root outputs/bc/alfworld/ctx=20,data=train:r1-gen-0.5-pred_reward/flan_t5/eval/dev/aggregation=mean
```

We can iterate this process by then generating rollouts with this new `r1` model, using the trained LFM to identify desirable behaviour, then training a `r2` model and so on.


## Adaptation

To adapt the model to the new environment, instead of generating rollouts on the training envs, we generate rollouts on the test envs, then use the trained LFM to extract desirable behaviour:

```
python evaluate_bc.py env=alfworld train_data=train eval.split=dev
python train_lfm.py env=alfworld p_pos=0.5 "infer_feedback='$PWD/outputs/bc/alfworld/ctx=20,data=train/flan_t5/eval/dev/aggregation=mean/'" p_pos=0.0 eval_name=adapt1
```

We can then improve the model on the test environment using LFM feedback:

```
python build_generated_feedback_finetune_data.py --inp_dir=outputs/reward/alfworld/intent/default_0.5/inferred_feedback/adapt1 --env=alfworld --strategy pred_reward --p_pos 0.5 --expname adapt1-
python train_bc.py env=alfworld train_data=train:adapt1-gen-0.5-pred_reward
```
