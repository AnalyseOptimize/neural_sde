# Neural SDE

Репозиторий сравнивает способы обучения SDE на симулированных траекториях.
Общий формат данных — `(batch, time, dimension)`, сетка времени — `(time,)`.

- `src/simulators/processes.py`: ABM, GBM, OU, детерминированный процесс, шум наблюдений и скачки.
- `src/nets/`: MLE (`sde_ml.py`), Latent SDE, SDE-GAN и SDE matching.
- `src/train.py`: циклы обучения, оценка качества, checkpoints.
- `scripts/main_train_*.py` и `configs/*.yaml`: точки входа Hydra и параметры экспериментов.
- `utils/visual.py`: графики; `utils/calibration.py`: восстановление функций коэффициентов OU.
- `logs/<run_name>_<timestamp>/`: параметры, история, сообщения, графики и веса запуска.

Установка: `python -m pip install -r requirements.txt`. Ниже команды используют
существующее окружение `.venv`. Запускать из корня репозитория.

## OU и MLE

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg \
  .venv/bin/python scripts/main_train_sde_ml.py
```

По умолчанию используется

\[
 dX_t = (0-X_t)\,dt + 0.3\,dW_t,\qquad X_0=2,\quad t\in[0,10].
\]

Среднее равно `2 exp(-t)`: сначала процесс приходит к нулю, затем колеблется
вокруг него со стационарным стандартным отклонением около `0.212`.
Симулятор генерирует точные переходы OU с шагом `0.02`, без скачков и шума
наблюдений. Обучаются две MLP от `(t, x)` с двумя скрытыми слоями по 16
нейронов: дрейф и положительная диффузия. Всего 674 параметра, 4096
траекторий, 1500 обновлений Adam. Форма OU при обучении не задаётся.

Файлы запуска:

- `config.yaml`, `history.json`, `messages.txt`, `checkpoints/model.pt`.
- `figs/simple_{drift,diffusion}_slices.pdf`: 40 срезов по времени/состоянию,
  непрерывные цветовые шкалы вместо легенд; пунктир — истинная функция OU.
- `figs/ou_coefficient_maps.{pdf,png}`: истина, оценка и ошибка на сетке
  `(t, x)`, 40 цветовых уровней. Пунктир ограничивает 2–98% новых траекторий.
- `coefficient_metrics.json`: RMSE на 512 независимых траекториях и отдельно
  на всём прямоугольнике, включая области экстраполяции.
- `coefficient_grids.npz`: численные значения функций и границы области данных.
- `figs/coupled_real_vs_generated_paths.pdf`: траектории на одном броуновском
  шуме; здесь и симулятор, и модель интегрируются схемой Эйлера.

MLE использует гауссовское Euler likelihood. При точных данных OU его
популяционные цели на конечном шаге отличаются от непрерывных коэффициентов:
`theta_eff = (1-exp(-theta*dt))/dt`,
`sigma_eff = sigma*sqrt((1-exp(-2*theta*dt))/(2*theta*dt))`.
Отчёт показывает ошибки относительно обеих целей. Поле
`learned_linear_projection` — линейная аппроксимация **уже обученной MLP**,
а не отдельно обученная параметрическая модель OU.

## Latent SDE

Основа — [Li et al., Scalable Gradients for Stochastic Differential Equations
(2020)](https://proceedings.mlr.press/v108/li20i.html) и примеры авторов
[`latent_sde.py`](https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py),
[`latent_sde_lorenz.py`](https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py).

Модель условная по наблюдаемому `x0`, как остальные генераторы репозитория.
Небольшой обратный GRU кодирует всю траекторию; нормированное время входит
в его вход, поэтому допустима и неравномерная сетка. Prior drift `h(z)` —
аффинный слой; posterior drift `f(z, context) = h(z) + MLP(z, context)` —
один скрытый слой с tanh. Диффузия — обучаемый положительный постоянный
диагональный вектор, общий для prior и posterior. Это намеренно ограниченная
параметризация: prior описывает линейную латентную динамику.

Обучение минимизирует отрицательный reconstruction log likelihood плюс KL:

\[
 -\mathbb E_q\sum_{i=1}^{T-1}\log p(x_i\mid z_i,z_0,x_0)
 +\beta\left[\mathrm{KL}(q(z_0)\Vert p(z_0))
 +\mathbb E_q\int\tfrac12\|(f-h)/g\|^2dt\right].
\]

Path-KL интегрирует `torchsde` с `logqp=True`. `beta` линейно увеличивается
до единицы. Для adjoint контекст явно включён в `adjoint_params`, чтобы
градиенты достигали GRU. Начальное наблюдение обусловлено, поэтому в
likelihood суммируются только последующие точки.

Доступны независимые `model.encoder_type` и `model.decoder_type`:

| Компонент | `deterministic` | `stochastic` |
| --- | --- | --- |
| Энкодер | Общее начальное состояние prior/posterior `z0=Linear(x0)`; начальный KL равен нулю | Gaussian `q(z0 \| trajectory)` и `p(z0 \| x0)`, reparameterization и аналитический KL |
| Декодер | Линейное среднее; фиксированный `train.noise_std` в reconstruction likelihood, без дополнительного observation noise при генерации | Линейное среднее и `softplus(Linear(z)) + decoder_std_min`; обучаемый Gaussian likelihood и сэмплирование observation noise |

Среднее декодера задаётся как `x0 + Linear(z_t - z0)`: сгенерированная
траектория начинается точно в заданной точке. Броуновский шум сохраняется
в обоих режимах. Детерминированный энкодер не заменяется дельта-распределением
против гауссовского prior: обе начальные меры совпадают, поэтому KL корректен.

```bash
# Детерминированные энкодер и декодер.
.venv/bin/python scripts/main_train_latent_sde.py --config-name latent_sde_deterministic

# Стохастические энкодер и декодер.
.venv/bin/python scripts/main_train_latent_sde.py --config-name latent_sde_stochastic

# Смешанный вариант, как в стандартном примере torchsde:
# Gaussian z0 и детерминированный readout с фиксированной шириной likelihood.
.venv/bin/python scripts/main_train_latent_sde.py
```

`sample_paths(ts, y0)` генерирует из prior. `reconstruct_paths(paths, ts)`
использует posterior с контекстом наблюдений. `sample_observations=False`
позволяет получить только среднее стохастического декодера для заданной
латентной траектории. Скрипт сохраняет оба вида графиков отдельно, а в
checkpoint — конфигурацию модели для восстановления через
`ConditionalLatentSDE.from_config(LatentSDEConfig(**checkpoint['model_config']))`.

Латентные коэффициенты зависят от выбора координат и не равны коэффициентам
наблюдаемого OU. Для прямой проверки восстановления `f(t,x), g(t,x)` служит
эксперимент MLE выше.

## Проверки

```bash
# При необходимости: .venv/bin/python -m pip install pytest
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg .venv/bin/python -m pytest -q
```

Тесты проверяют KL, градиенты всех комбинаций энкодера/декодера с обычным
solver и adjoint, условное начальное значение, observation noise,
checkpoint, моменты OU и совпадение векторизованного MLE с пошаговым
вычислением, включая градиенты.
