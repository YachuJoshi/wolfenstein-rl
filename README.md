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
python3 ./main.py --level { basic | defend | deadly } --train
```

### 2. For Testing:

```bash
python3 ./main.py --level { basic | defend | deadly } --test --steps { steps }
```

## Valid Level Mode Command:

1. --level basic --train
2. --level basic --test --steps { steps }
3. --level defend --train
4. --level defend --test --steps { steps }
5. --level deadly --train
6. --level deadly --test --steps { steps }

## Examples:

### 1. To _train_ the model on basic level:

```bash
python3 ./main.py --level basic --train
```

### 2. To _test_ the model trained for _200000_ steps on _DEFEND_ level:

```bash
python3 ./main.py --level defend --test --steps 200000
```
