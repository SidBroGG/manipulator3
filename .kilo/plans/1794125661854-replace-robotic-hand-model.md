# Замена модели манипулятора на RoboticHandMain из temp

## Цель

Заменить текущую модель руки (`assets/root.urdf`, `assets/manipulator.xml`, `assets/meshes/*.stl`) на новый CAD-экспорт `temp/RoboticHandMain_urdf`, сохранив внешний интерфейс `revolute_1..7`, роли `base_link / base_spinning / shoulder_link / elbow_link / wrist_link / main__gripperbase / main__gripper1/2`, протокол Zenoh и имена аппаратных суставов.

После замены должны проходить все тесты, включая полный pick-and-place цикл в MuJoCo.

## Принятые решения

- **Имена**: внешние имена суставов/звеньев сохраняются. Новые имена (`Base_Revolute`, `Motor1_Telo94`, `Claw_Revolute` и т.д.) используются только как источник геометрии и преобразований.
- **Сенсоры**: `gripper_camera` и `tof_site` крепятся к `main__gripperbase`, который является виртуальным кадром сенсора, зафиксированным на `servo_lower` (аналог `wrist_mount`), а не на поворотном основании клешни. `revolute_5` (roll клешни) остаётся в 5-суставном интерфейсе, но не влияет на позу сенсора.
- **Клешня**: `main__gripper1/2` — прямые шарниры челюстей вокруг штатных `joint_gear_to_claw_*`. Шестерни и коннекторы остаются визуальными, без коллизий. `revolute_6` мимикрирует `revolute_7`.
- **Старые ассеты** удаляются; `temp/` остаётся в `.gitignore`.

## 1. Подготовка ассетов

- [ ] Скопировать все `RoboticHandMain-*.stl` из `temp/RoboticHandMain_urdf/` в `assets/meshes/` (имена сохранить как в экспорте, чтобы не потерять соответствие CAD).
- [ ] Не копировать `RoboticHandMain.sc`, `RoboticHandMain.sc.bak`, `RoboticHandMain.urdf` в репозиторий: исходный URDF переписывается вручную, `.sc` не нужен.
- [ ] Удалить старые меши: `base_stand.stl`, `base_spinning.stl`, `shoulder.stl`, `elbow.stl`, `wrist.stl`, `gripper_base.stl`, `gripper_finger1.stl`, `gripper_finger2.stl`.
- [ ] Добавить новые меши в git (они не должны попасть под `temp/`).

Соответствие новых мешей и ролей:

| Новый меш | Роль в проекте |
|---|---|
| `RoboticHandMain-base_link.stl` | статическое основание `base_link` |
| `RoboticHandMain-link_1.stl` | вращающееся основание `base_spinning` |
| `RoboticHandMain-link_2.stl` | `shoulder_link` |
| `RoboticHandMain-link_3.stl` | `elbow_link` |
| `RoboticHandMain-wrist_mount.stl` | `wrist_link` |
| `RoboticHandMain-servo_lower.stl` | `servo_lower` (фиксированное звено под сенсором) |
| `RoboticHandMain-claw_base.stl`, `RoboticHandMain-claw_base-collision.stl` | `claw_base` (visual + collision) |
| `RoboticHandMain-claw_lower.stl`, `RoboticHandMain-claw_upper.stl` | `main__gripper1/2` |
| `RoboticHandMain-gear_lower.stl`, `RoboticHandMain-gear_upper.stl` | визуальные шестерни на fixed-кадрах |
| `RoboticHandMain-connector_lower.stl`, `RoboticHandMain-connector_upper.stl` | визуальные коннекторы |
| `RoboticHandMain-mg996R_v2_step*.stl` | визуальные корпуса сервоприводов |
| `RoboticHandMain-mg996_arm_6pad_step*.stl` | визуальные качалки сервоприводов |
| `RoboticHandMain-RPi_cam_Rev_1_3_SLDPRT__1_.stl` | визуальная камера на `wrist_link` |
| `RoboticHandMain-____________*.stl` (3 файла) | визуальные детали на `base_spinning` |

## 2. Новый `assets/root.urdf` (кинематика для IKPy)

Переписать URDF из нового экспорта, оставив только структурные звенья и нужные fixed-кадры. Декоративные сервы/качалки/шестерни/коннекторы в URDF не включать — они нужны только в MJCF.

### Состав URDF

Звенья:

- `base_link`
- `base_spinning`
- `shoulder_link`
- `elbow_link`
- `wrist_link`
- `servo_lower`
- `main__gripperbase` (виртуальный кадр сенсора, без меша)
- `claw_base`
- `gear_lower`
- `gear_upper`
- `main__gripper1`
- `main__gripper2`

Суставы:

- `revolute_1`: `base_link` → `base_spinning` (новый `Base_Revolute`)
- `revolute_2`: `base_spinning` → `shoulder_link` (новый `Motor1_Telo94`)
- `revolute_3`: `shoulder_link` → `elbow_link` (новый `Motor3_Link3`)
- `revolute_4`: `elbow_link` → `wrist_link` (новый `Motor4_Wrist`)
- `servo_lower_fixed`: `wrist_link` → `servo_lower` (новый `ServoLower_Wrist`)
- `main__gripperbase_fixed`: `servo_lower` → `main__gripperbase` (origin = origin нового `Claw_Revolute`)
- `revolute_5`: `servo_lower` → `claw_base` (новый `Claw_Revolute`, тот же origin)
- `gear_lower_fixed`: `claw_base` → `gear_lower` (origin нового `joint_servo_gear`)
- `gear_upper_fixed`: `claw_base` → `gear_upper` (origin нового `joint_base_gear_upper`)
- `revolute_7`: `gear_upper` → `main__gripper1` (origin нового `joint_gear_to_claw_upper`)
- `revolute_6`: `gear_lower` → `main__gripper2` (origin нового `joint_gear_to_claw_lower`, `<mimic joint="revolute_7" multiplier="-1.0">`)

Порядок в XML важен для IKPy: после `wrist_link` первым должен идти `servo_lower_fixed`, после `servo_lower` первым — `main__gripperbase_fixed`, чтобы обход остановился на сенсоре и не ушёл в клешню.

### Нормализация осей и поз

- **Не поворачивать основание на 180°**, как в старой модели. Новый CAD при `q1=0` и мировом yaw `0` уже наклоняет руку к +X при отрицательных `q2/q3`: ось качания плеча получается `-Y`, как в старой семантике. В MJCF основание: `pos="0 0 BASE_HEIGHT_WORLD" euler="0 0 0"`.
- Все `origin xyz/rpy`, `visual/collision origin` и `inertial` структурных звеньев брать из нового URDF **без изменений**. Не менять origin первого сустава: визуальный origin `link_1` (`rpy=-π/2`) компенсирует `Base_Revolute` (`rpy=+π/2`), и при `q1=0` вращающееся основание совпадает с CAD.
- Единственная нормализация первого сустава — axis `(0, 0, -1)` вместо `(0, 0, 1)`, чтобы сохранить старую семантику `q1 = -azimuth`. Ось можно инвертировать, не трогая origin и меши.
- Оси `revolute_2..4` оставить локальными `(0, 0, 1)` с их URDF-ориентациями. Новые лимиты из экспорта: `[-π/2, 0]` для `revolute_2`, `revolute_3`, `revolute_4`.
- Ожидаемая замена знаков: `q2`, `q3` остаются отрицательными (как в старых позах), `q4` (кисть) меняет знак: старые `+0.9 / +0.827` становятся `-0.9 / -0.827`.
- `revolute_1`: лимит `[-π, π]` (как у stepper).
- `revolute_5`: лимит `[-π/2, π/2]` (как у сервопривода roll); знак roll не критичен, все baseline-позы используют `0`.
- `revolute_6/7`: лимиты подобрать по реальному раскрытию новой клешни; стартовая точка `[-0.6, 0.6]`.
- Для `main__gripperbase` использовать маленькую инерцию `1e-6`, как для служебных звеньев старой модели.
- Пути к мешам: `meshes/RoboticHandMain-*.stl`.
- Если калибровка покажет, что при yaw `0` рука уходит от +X, допускается повернуть всё основание на 180° и синхронно поменять знак `q1`/оси, но не менять origin первого сустава в отрыве от мешей.

### Начальные baseline-позы

Старые значения использовать как стартовую точку, но у `revolute_4` ожидается инверсия знака из-за другой ориентации оси:

```python
POSE_HOME = [0.0, 0.0, 0.0, 0.0, 0.0]
POSE_SCAN = [0.0, -0.1364, -1.2642, -0.9000, 0.0]
POSE_DROP_OFF = [-1.6814, -0.6843, -1.4092, -0.8270, 0.0]
```

Значения обязательно проверить через FK/IK и подстроить по критериям из раздела 5; тесты не должны молча подгоняться под неверную позу.

## 3. Новый `assets/manipulator.xml`

Собрать MJCF вручную по иерархии нового URDF, сохранив стиль текущего файла.

### Иерархия

```
base_link
└── base_spinning (revolute_1)
    └── shoulder_link (revolute_2)
        └── elbow_link (revolute_3)
            └── wrist_link (revolute_4)
                └── servo_lower (fixed)
                    ├── main__gripperbase (fixed, sensor frame; camera + tof_site)
                    └── claw_base (revolute_5)
                        ├── gear_lower (fixed)
                        │   └── main__gripper2 (revolute_6)
                        └── gear_upper (fixed)
                            └── main__gripper1 (revolute_7)
```

### Геометрия и физика

- Для структурных звеньев: `visual` + `collision` из новых мешей; для `claw_base` использовать отдельный `claw_base-collision.stl`.
- Челюсти `main__gripper1/2` — класс `gripper_finger` (трение как у старых губок).
- Все декоративные сервы, качалки, шестерни, коннекторы, RPi-камера и три безымянные детали — только visual-геомы (`contype=0`, `conaffinity=0`), привязанные к ближайшему каноническому body с их URDF visual origin. Пустые звенья (`RoboticHandMain`, `link_4`, `link_5`, `Testpart_step`, `Nema17`, винты) не переносить.
- Инерцию не задавать явно для звеньев с мешами — следовать текущей MJCF-конвенции (density=1050), чтобы не получить безмассовые `link_3`/`wrist_mount` из экспорта. Для `main__gripperbase` — маленькая явная инерция.
- Актуаторы: `act_revolute_1..5`, `act_gripper`; ctrlrange по лимитам; kp/kv начать со старых значений и подстроить при нестабильности.
- Equality: `revolute_6 = -revolute_7`.
- Контакты-исключения: `base_link/base_spinning`, `main__gripper1/main__gripper2`, `main__gripper1/claw_base`, `main__gripper2/claw_base`.

### Сенсорный риг

- `gripper_camera` на `main__gripperbase`, оптическая ось вдоль `+Z` кадра сенсора; сохранить OV5647 FOV `fovy=48.8`. Стартовые `xyaxes` взять из старой модели (`0 -1 0 -1 0 0`), но проверить ориентацию изображения: кадр сенсора нового CAD может быть повёрнут, при необходимости скорректировать `xyaxes`/quat так, чтобы «верх» кадра соответствовал старой модели.
- `tof_site` на `main__gripperbase`, направление `+Z`; стартовый offset по Z `0.048` м, как раньше.
- Проверить, что луч ToF не попадает в собственные геомы новой клешни. Если попадает — либо вынести site вперёд за челюсти, либо изменить маску `get_tof_reading` (робот в отдельной geom group, луч только по группе стола/объектов). Критерий: в HOME `get_tof_reading() == -1.0`, при наклоне к столу расстояние уменьшается.

## 4. Правки Python-кода

### `src/manipulator3/model.py`

- `load_ikpy_chain` должен возвращать цепочку до сенсора:
  `OriginLink + revolute_1..4 + servo_lower_fixed + main__gripperbase_fixed` = 7 звеньев.
  Маска: `[False, True, True, True, True, False, False]`.
- Убрать срез `links[:6]` в `KinematicsEngine`.
- Обновить docstring: `main__gripperbase` — кадр сенсора, а не физическое основание клешни.

### `src/manipulator3/kinematics.py`

- `KinematicsEngine`:
  - IKPy-цепочка из `load_ikpy_chain` целиком, без пересоздания с 5 активными суставами.
  - Для IK использовать только `revolute_1..4` (`ik_joints`, `ik_dofs`, `ik_q_adrs`, `ik_limits`).
  - `revolute_5` сохранять из `initial_guess` (или 0.0) и не менять в DLS; в якобиан не включать.
  - Выход IK — по-прежнему массив из 5 углов: 4 решённых + текущий roll.
  - `forward_kinematics` по-прежнему пишет все 5 qpos; поза сенсора от roll не зависит.
- `POSE_*` и `joint_limits` обновить по результатам калибровки (раздел 5).
- `BASE_HEIGHT_WORLD` синхронизировать с MJCF-позицией основания.
- `world_to_base_pos` / `base_to_world_pos` упростить до чистого сдвига по Z (без поворота на 180°), так как новое основание ставится с `euler=0`:
  `pos_base = [x, y, z - BASE_HEIGHT_WORLD]`, `pos_world = [x, y, z + BASE_HEIGHT_WORLD]`.
- Проверить `heuristic_yaw = -arctan2(y, x)` и `SectorScanner` после смены axis `revolute_1` на `-Z`.

### `src/manipulator3/servoing.py`

- `JOINT_PITCH=3` сохранить, но `DEFAULT_JOINT_LIMITS` привести к новым лимитам `revolute_4` (`[-1.5708, 0]`).
- Проверить знак `u_pitch`: после инверсии оси кисти он может требовать противоположного знака. Настроить по тесту визуального серво и E2E.
- `SectorScanner.generate_sweep_waypoints`: знаки `tilt_offsets` подобрать так, чтобы камера сканировала стол выше/ниже линии SCAN. В `state_machine.py` обновить `tilt_offsets=[...]` синхронно.

### `src/manipulator3/grasping.py`

- После калибровки обновить `GRIPPER_OPEN`, `GRIPPER_GRASP_CMD`, `GRIPPER_CLOSED_EMPTY`, `GRIPPER_CONTACT_MIN/MAX_RAD`.
- Обновить `TOF_TARGET_MIN/MAX/NOMINAL_MM` и `TABLE_CLEARANCE_MIN_EE_Z` под новую геометрию челюстей и камеры.
- Перепроверить знаки «арки» в `plan_transport`/`plan_return_home` (индексы 1/2/3): после инверсии знака кисти компенсация `positions[:, 3] -= 0.65 * arch` может требовать противоположного знака, иначе кисть будет задираться вверх по пути.
- Знак команд оставить старым: положительное открывает, отрицательное закрывает. Если новый шарнир челюсти работает наоборот, инвертировать ось в URDF/MJCF, а не размазывать инверсию по коду.

### `src/manipulator3/hardware.py`

- Пины и имена не менять.
- Если лимиты новых суставов отличаются от `ServoConfig.min/max`, обновить диапазоны и добавить пометку, что физическая калибровка нулей/направлений должна быть перепроверена на железе.

### `src/manipulator3/verify_model.py`

- Обновить список ожидаемых body/joint/actuator.
- Проверять камеру и ToF на `main__gripperbase`.
- Проверять mimic `revolute_6 = -revolute_7`.
- Печатать позицию сенсора в HOME и результат ToF.

### Примеры

- `examples/sim_preview.py`, `examples/zenoh_preview.py`: заменить жёстко заданные положительные углы кисти на новые отрицательные/калиброванные.
- `examples/statemachine_preview.py`, `examples/detector_preview.py`, `examples/kinematics_preview.py`: проверить после калибровки, поправить только при необходимости.

## 5. Калибровка (обязательный этап)

Создать временный скрипт `temp/calibrate_new_model.py` (в git не добавлять, `temp/` уже в `.gitignore`). Он загружает `assets/scene.xml`, умеет ставить q, делать `mj_forward`, печатать позиции/матрицы тел, рендерить камеру и мерить ToF.

Порядок:

1. **Высота основания**: вычислить min Z вершин меша `base_link`; подобрать `BASE_HEIGHT_WORLD` и `pos` основания так, чтобы низ стоял на столе Z=0. Синхронизировать `kinematics.py` и `world_to_base_pos` (сдвиг по Z).
2. **Проверка осей/знаков**: убедиться, что при `q1=0` плоскость качания содержит +X, `q2<0` наклоняет руку к +X, `q1=-azimuth` поворачивает к азимуту. Если рука уходит от +X — повернуть основание на 180° и синхронно скорректировать axis/знаки, не меняя origin первого сустава отдельно от мешей.
3. **HOME/SCAN/DROP_OFF**: через FK/IK получить позы, при которых:
   - HOME: рука вертикально, сенсор `Z > 0.3`, ToF чист;
   - SCAN: камера смотрит на стол в районе `(0.18, 0, 0)`, сенсор над столом с безопасным зазором;
   - DROP_OFF: сенсор над зоной выгрузки `(-0.02, 0.18)`, ось подхода вниз.
   Записать итоговые 5-векторы в `kinematics.py`.
4. **Клешня**: определить диапазон и знак `revolute_6/7`; выставить лимиты; подобрать `GRIPPER_OPEN`, `GRASP`, `CLOSED_EMPTY` и порог `verify_grip` так, чтобы куб 30 мм удерживался, а пустая клешня не давала ложного контакта.
5. **Камера**: на `POSE_SCAN` отрендерить `gripper_camera`; добиться, чтобы куб был виден в центре, а новые губки — по краям кадра (как в старой модели). При необходимости подстроить offset камеры, но не FOV. Если новые губки видны не у боковых краёв, скорректировать `gripper_margin_ratio`/`is_gripper_finger` в `detector.py`, чтобы детектор не принимал их за цель.
6. **ToF**: на HOME — `-1.0`; при наклоне к столу расстояние монотонно падает; в точке захвата соответствует положению губок вокруг объекта. Подстроить offset site и `TOF_TARGET_*`.
7. **E2E**: прогнать state machine; при сбое на визуальном серво/подходе/захвате вернуться к шагам 2–6.

## 6. Тесты

Обновить существующие stage-тесты под новую модель:

- `tests/test_stage1_model.py`
  - IKPy chain: 7 звеньев, маска `[False, True, True, True, True, False, False]`.
  - Ожидаемые body: `base_link`, `base_spinning`, `shoulder_link`, `elbow_link`, `wrist_link`, `servo_lower`, `main__gripperbase`, `claw_base`, `main__gripper1/2`.
  - Камера/ToF на `main__gripperbase`, FOV 48.8.
  - Mimic: `q6 ≈ -q7`; команда открытия должна открывать челюсти.
  - FK HOME: `Z > 0.3`; смещённая поза — в новых лимитах.
- `tests/test_stage2_simulation.py`
  - Заменить старые положительные углы кисти на новые калиброванные.
  - Проверка ToF: в HOME `-1.0`, при наклоне расстояние падает.
  - Проверка устойчивости основания (min Z меша = 0).
- `tests/test_stage4_kinematics.py`
  - FK-эквивалентность: IKPy `[0, q1, q2, q3, q4, 0, 0]` против MuJoCo с произвольным `q5`.
  - Отдельный тест: изменение `revolute_5` не меняет позу `main__gripperbase` (сенсор развязан с roll).
  - IK grid и baseline poses — в новых лимитах; DROP_OFF над зоной выгрузки.
  - `world_to_base_pos`/`base_to_world_pos`: проверить, что это чистый сдвиг по Z и что FK-эквивалентность держится.
- `tests/test_stage6_detector.py`
  - `q_init[3]` и знаки ошибки кисти привести к новой модели; тест визуального серво должен сходиться.
- `tests/test_stage7_statemachine.py`
  - Обновить константы захвата/ToF; E2E должен давать смещение куба > 0.08 м.
- `tests/test_stage3_zenoh.py`, `test_stage8_thin_client.py`, `test_stage9_servo_tester.py` — проверить, что не зависят от старых лимитов; имена сохранены, правок быть не должно.

Не добавлять избыточные тесты на тривиальный код; новые тесты только на критичные инварианты (FK-эквивалентность, развязка сенсора и roll, ToF self-hit, симметрия губок).

## 7. Валидация

Выполнить по порядку:

```powershell
uv run python -m manipulator3.verify_model
uv run pytest -q
uv run python examples/sim_preview.py
uv run python examples/kinematics_preview.py
uv run python examples/statemachine_preview.py
```

Критерии приёмки:

- [ ] `verify_model` проходит, камера/ToF на `main__gripperbase`, mimic работает.
- [ ] Все тесты зелёные, включая E2E pick-and-place.
- [ ] В `sim_preview` видна новая рука и новая клешня, камера показывает стол, ToF уменьшается при подводе.
- [ ] `HOME/SCAN/DROP_OFF` плавно достигаются, нет выхода за лимиты и продавливания стола.
- [ ] В репозитории не осталось ссылок на старые меши/старую геометрию, `temp/` не добавлен в git.
- [ ] После проверки — commit с осмысленным сообщением и `git push origin main` (по правилам проекта).

## Риски и границы

- **Главный риск** — численная калибровка поз, знаков кисти, лимитов и порогов захвата. Все значения из раздела 4 — стартовые; их нужно подтвердить FK/IK и тестами.
- **Конверсия URDF rpy → MJCF**: выполнять через кватернионы (`quat`), а не через `euler`, чтобы не перепутать порядок осей. FK-эквивалентность MuJoCo/IKPy — обязательный тест.
- **Ориентация основания**: новый CAD не обязан повторять старое `euler=180°`. Стартовое решение — `euler=0` и axis `revolute_1=-Z`; если калибровка покажет неверное направление, повернуть всё основание и поправить `world_to_base_pos` вместе с тестами, не меняя origin сустава отдельно от мешей.
- **ToF self-hit**: новые губки/шестерни могут попадать в луч. Решение — offset site или отдельная geom group для робота.
- **Физическая калибровка**: нули и направления сервоприводов старой руки могут не совпадать с новой. Реальную руку не запускать до проверки `ServoConfig` и нулей; это отдельная задача Stage 9.
- **Четырёхзвенник клешни** не моделируется точно: шестерни/коннекторы визуальные, челюсти — прямые шарниры. Это осознанное упрощение, согласованное с пользователем.
- **Camera/ToF**: виртуальный сенсор стоит в кадре основания клешни; латеральное смещение реальной RPi-камеры не моделируется до hand-eye калибровки Stage 9.1.

## Чеклист для пользователя после реализации

- [ ] Запустить `uv run python -m manipulator3.verify_model` и `uv run pytest -q`.
- [ ] Открыть `examples/sim_preview.py`: проверить, что форма руки/клешни соответствует новой модели, камера и ToF работают.
- [ ] Запустить `examples/statemachine_preview.py`: куб должен быть найден, захвачен и перенесён в зону выгрузки.
- [ ] На железе: перепроверить нули/направления всех сервоприводов и лимиты `revolute_2..5`; при необходимости обновить `ServoConfig` и только потом запускать thin client.
