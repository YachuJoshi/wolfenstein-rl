# Self Learning Agent in FPS Environment - RL

Simple implementation of **RayCasting Algorithm** integrated with _Proximal Policy Optimization ( PPO )_ Deep RL Learning Technique.

## Levels:

1. Basic
2. Defend The Center
3. Deadly Corridor

## Modes:

1. Train ( Training The Model )
2. Test ( Testing The Model )

## CLI Commands:

### 1. For Training:

```bash
python3 ./main.py {level}-train
```

### 2. For Testing:

```bash
python3 ./main.py {level}-test {steps}
```

## Valid Level-Mode:

1. basic-train
2. basic-test
3. defend-train
4. defend-test
5. deadly-train
6. deadly-test

## Examples:

### 1. To _train_ the model on basic level:

```bash
python3 ./main.py basic-train
```

### 2. To _test_ the model trained for _200000_ steps on _DEFEND_ level:

```bash
python3 ./main.py defend-test 200000
```
