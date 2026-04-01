# Sub-5nm EUV 리소그래피용 정밀유리 기판 Micro-Crack Lifecycle 관리 시뮬레이터 개발

**제안 기관:** 성균관대학교 SPMDL (권석준 교수 연구실)  
**협력 대상:** Corning Incorporated — 천안 사업장  
**제안 일자:** 2026년 2월  
**개정:** v2 — 2026년 3월 (Ideal Glass 이론적 프레임워크 추가)

---

## 1. 연구 배경 및 필요성

### EUV/High-NA 전환에 따른 substrate 정밀도 요구 급증

차세대 반도체 공정이 Sub-5nm 노드로 진입하면서, EUV 리소그래피 광학계 및 photomask substrate에 요구되는 정밀도가 급격히 높아지고 있습니다. ASML EXE:5200 (High-NA, 0.55 NA) 도입과 함께 overlay 오차 허용 범위가 **1.5 nm RMS 이하**로 축소되었으며, 이는 glass substrate의 열적·기계적 안정성에 대한 전례 없는 수준의 요구입니다.

### Micro-crack: Overlay 오차와 yield 저하의 핵심 원인

Glass substrate 내부의 micro-crack은 다음 경로를 통해 공정 성능을 저하시킵니다:

- **국소적 CTE(열팽창계수) 이상** → 열 변형 → overlay registration 오차
- **국소적 굴절률 변화 (Δn)** → 위상 오차 → pattern placement 오차
- **Subcritical crack growth (SCG)** → 사용 중 균열 진전 → 점진적 성능 열화

현재 업계에서는 **사후 검사(post-mortem inspection)** 위주의 품질 관리가 이루어지고 있어, 균열의 발생 시점과 진전 경로를 예측하기 어렵습니다. 이는 예기치 않은 공정 중단과 yield 손실로 이어집니다.

### Corning ULE 7973의 기술적 과제

Corning ULE 7973은 EUV photomask substrate의 사실상 표준(de facto standard)이나, 비정질(amorphous) 구조 특성상 결정질 소재 대비 **균열 거동 예측이 본질적으로 어렵습니다**. 결정 입계(grain boundary)가 없어 crack deflection이 일어나지 않으며, subcritical crack growth 거동이 환경 조건(습도, 온도)에 민감합니다.

<!-- ==================== v2 신규 추가: Ideal Glass 섹션 시작 ==================== -->

### Ideal Glass와 Micro-Crack Nucleation: 새로운 이론적 프레임워크

최근 Bolton-Lum 등(2025)은 polydisperse 디스크 시스템에서 transient degrees of freedom을 활용하여 **2D ideal glass**를 구현하는 데 성공하였습니다 [7]. Ideal glass란 비정질(amorphous) 구조를 유지하면서도 configurational entropy가 0에 수렴(S_config → 0)하여 **결정과 동등한 수준의 mechanical properties**를 보이는 이론적 극한 상태입니다.

이 결과는 EUV photomask substrate의 균열 거동 이해에 근본적인 시사점을 제공합니다:

1. **Crack nucleation site의 부재:** Ideal glass는 구조적 결함(dangling bond, void, density fluctuation)이 극소화된 상태입니다. 일반적인 비정질 유리에서 micro-crack의 nucleation은 이러한 구조적 결함이 응력 집중(stress concentration) 지점으로 작용하여 발생하는데, ideal glass에 가까울수록 이러한 nucleation site의 밀도가 본질적으로 감소합니다.

2. **Hyperuniformity에 의한 crack propagation 억제:** Bolton-Lum 등이 확인한 ideal glass의 핵심 특성 중 하나는 **hyperuniformity** — 장거리 밀도 요동(long-range density fluctuation)의 억제 — 입니다. 일반 유리에서 밀도 fluctuation은 국소적 탄성계수(local elastic modulus) 및 CTE의 불균일성을 야기하며, 이는 crack propagation의 취약 경로(weak path)를 형성합니다. Hyperuniform 구조에서는 이러한 장거리 불균일성이 원천적으로 억제되어 crack propagation의 directional bias가 사라집니다.

3. **"Ideal Glass Proximity" 지표의 실용적 가치:** ULE 7973가 ideal glass에 얼마나 근접하는지를 정량화하는 것은 sub-5nm 리소그래피 수율 예측의 핵심이 될 수 있습니다. 구체적으로, S_config의 잔여 크기, density fluctuation의 power spectrum, 그리고 hyperuniformity metric (장파장 structure factor S(q→0)의 수렴 속도) 등을 통해 ULE 7973의 "ideal glass proximity"를 정량화할 수 있습니다.

**본 시뮬레이터의 M1(Nucleation Engine)에서 이 "ideal glass proximity" 개념을 직접 반영할 수 있습니다.** 현재 M1의 Griffith criterion 기반 nucleation 확률 계산에서 결함 분포(Poisson/Neyman-Scott)의 파라미터를 ideal glass proximity metric과 연동하면, "이상적 유리에 가까운 기판일수록 nucleation 확률이 어떻게 감소하는가"를 정량적으로 시뮬레이션할 수 있습니다. 이는 Corning의 차세대 Extreme-ULE 개발 시 **목표 물성치 설정에 대한 이론적 근거**를 제공합니다.

<!-- ==================== v2 신규 추가: Ideal Glass 섹션 끝 ==================== -->

---

## 2. 연구 목표

본 과제는 **Glass Micro-Crack Lifecycle Simulator**를 개발하여 다음 세 가지 핵심 목표를 달성하고자 합니다:

1. **Full Lifecycle 시뮬레이터 개발**: Nucleation → Propagation → Inspection → Diagnostics → Attribution의 5단계를 통합하는 physics-informed 시뮬레이션 플랫폼 구축
2. **경쟁 소재 대비 ULE 7973 우위 정량화**: Zerodur, Clearceram-Z, AGC AZ, Shin-Etsu Quartz 등 5개 소재와의 체계적 비교를 통해 ULE 7973의 기술적 moat를 수치적으로 입증
3. **ML 기반 실시간 진단 시스템 프로토타입**: Bayesian inference 기반의 역추론 엔진을 통해 검사 데이터로부터 균열 상태를 실시간 진단하는 프로토타입 시스템 개발
<!-- v2 추가: 4번 목표 -->
4. **Ideal Glass Proximity 정량화 프레임워크**: ULE 7973의 비정질 구조가 ideal glass 극한에 얼마나 근접하는지를 정량화하고, 이를 nucleation 확률 및 crack propagation 거동 예측에 반영하는 이론적 프레임워크 구축

---

## 3. 핵심 기술 및 방법론

본 시뮬레이터는 5개의 독립 모듈로 구성되며, 각 모듈은 물리 법칙에 기반한 forward model과 데이터 기반 inverse model을 결합합니다.

> **[Fig. 0 참조: 시스템 구조도 — fig0_schematic.png]**

| 모듈 | 기능 | 핵심 물리/알고리즘 |
|------|------|-------------------|
| **M1: Nucleation Engine** | 결함 분포 생성 및 nucleation 확률 계산 | Griffith criterion, Monte Carlo, Poisson/Neyman-Scott 결함 분포, <!-- v2 추가 --> Ideal glass proximity metric 연동 |
| **M2: Propagation Engine** | Subcritical crack growth 및 phase-field 균열 진전 | Charles-Hillig velocity law, Paris fatigue law, Phase-field fracture |
| **M3: Inspection Forward Model** | 6종 검사 기법의 신호 시뮬레이션 | Lamb wave dispersion, Rayleigh/Mie scattering, Raman stress mapping, 193nm interferometry, EELS, KFM |
| **M4: Inverse ML Diagnostics** | 검사 신호로부터 균열 상태 역추론 | Gaussian Process Classifier/Regressor, Bayesian inference, Physics-informed features |
| **M5: Process Attribution** | 공정 분산 분해 및 교체 시점 최적화 | Variance decomposition, Bayesian changepoint detection, Cost optimization |

현재 **M1–M5 전체 모듈 구현이 완료**되었으며 (156/156 unit tests 통과), 시뮬레이션 기반 검증을 진행 중입니다.

<!-- v2 추가: M1 확장 설명 시작 -->

### M1 확장: Ideal Glass Proximity 기반 Nucleation 모델

기존 M1 모듈의 Griffith criterion 기반 nucleation 확률 계산을 다음과 같이 확장합니다:

- **Hyperuniformity metric 도입:** 장파장 structure factor S(q→0)의 수렴 속도를 기반으로 결함 분포의 장거리 균일성을 정량화합니다. Bolton-Lum 등 [7]이 제시한 ideal glass의 S(q) ~ q^α (α > 0) 스케일링을 참조 기준으로 사용합니다.
- **Configurational entropy proxy:** 실험적으로 측정 가능한 물성(잔류 응력 분포, Boson peak intensity, fictive temperature 분포)으로부터 S_config의 proxy를 구성하고, 이를 nucleation probability의 pre-factor에 반영합니다.
- **Density fluctuation power spectrum:** 기판 내 밀도 요동의 power spectrum을 입력받아, 국소적 stress concentration의 통계적 분포를 계산합니다. Ideal glass에 가까울수록 power spectrum의 저주파 성분이 억제되어 nucleation 확률이 체계적으로 감소합니다.

이 확장을 통해, **"ULE 7973의 물성을 어느 방향으로 개선하면 nucleation 확률이 가장 효과적으로 감소하는가"**에 대한 정량적 답을 제시할 수 있습니다.

<!-- v2 추가: M1 확장 설명 끝 -->

---

## 4. 기대 효과

시뮬레이션 기반 추정에 따르면, 본 시스템의 적용을 통해 다음과 같은 효과를 기대할 수 있습니다:

| 항목 | 추정 효과 | 근거 |
|------|----------|------|
| Overlay 오차 기여분 감소 | 15–30% (mask degradation 성분) | M5 variance decomposition 기반 |
| 비계획 정지 감소 | 20–40% | Bayesian changepoint에 의한 조기 경보 |
| Substrate 교체 비용 절감 | 연간 $100K–$500K (라인 당) | M5 replacement optimization 기반 |
| 공정 최적화 시간 단축 | 30–50% | M4 역추론에 의한 root cause 자동 진단 |
<!-- v2 추가 -->
| Extreme-ULE 개발 목표 물성 도출 | 개발 기간 단축 기대 | Ideal glass proximity 기반 nucleation 최적화 |

> ※ 상기 수치는 시뮬레이션 기반 추정치이며, 실제 효과는 공정 조건과 운영 환경에 따라 달라질 수 있습니다.

**Corning 관점의 전략적 가치:**
- ULE 7973의 SCG exponent (n=20), 높은 활성화 에너지 (80 kJ/mol) 등 경쟁 우위를 **정량적으로 입증**하는 도구 확보
- 고객사(TSMC, Samsung, Intel 등)에 대한 **기술 차별화 자료** 제공
- High-NA EUV 전환 시 Extreme-ULE 개발 방향에 대한 **시뮬레이션 기반 가이드라인** 제시
<!-- v2 추가 -->
- **Ideal glass proximity 프레임워크**를 통해 ULE 7973의 구조적 완전성을 이론적 극한과 비교하는 정량적 벤치마크 확보 — 이는 경쟁사 대비 차별화된 기술 내러티브를 제공

---

## 5. 연구진 및 수행 체계

### 연구 책임자
- **권석준 교수** — 성균관대학교 SPMDL (Semiconductor Physics & Materials Design Lab)

### 수행 일정

| Phase | 기간 | 주요 내용 |
|-------|------|----------|
| **Phase 1** | 8개월 | 시뮬레이터 고도화 및 Corning 실험 데이터 연동 (M1–M3 calibration) |
| **Phase 2** | 10개월 | ML 진단 시스템 구축 및 현장 실증 (M4 transfer learning, M5 실공정 검증) |
| **Phase 3** | 6개월 | 통합 플랫폼 완성 및 기술 이전 |

### 협력 체계
- **Corning 천안:** 실험 데이터 제공, ULE 7973 시편, 현장 검증 지원
- **SPMDL:** 시뮬레이터 개발, ML 모델링, 데이터 분석
- **Advisory Board:** 분기별 phase-gate 리뷰를 통한 연구 방향 관리

---

## 6. 시뮬레이션 결과 미리보기

아래 그림들은 현재 구현된 시뮬레이터의 대표적인 출력 결과입니다.

### Fig. 1 — Nucleation Probability Map (M1)
> **[fig1_nucleation_map.png]**  
> ULE 7973 기판(152×152 mm) 위의 결함 분포와 nucleation 확률 heatmap. CTE 불균일성에 의한 thermoelastic stress와 Griffith criterion 기반 nucleation 확률을 보여줍니다.

### Fig. 2 — Subcritical Crack Growth (M2)
> **[fig2_crack_growth.png]**  
> (a) 시간에 따른 균열 길이 성장 궤적. (b) Charles-Hillig V–K_I diagram으로 subcritical 영역(K_0 < K_I < K_IC)에서의 crack velocity를 보여줍니다.

### Fig. 3 — Inspection Method Comparison (M3)
> **[fig3_inspection_comparison.png]**  
> 6종 비파괴 검사 기법(Acoustic, Laser Scattering, Raman, 193nm Interferometry, EELS, KFM)의 sensitivity, resolution, depth penetration, speed, cost effectiveness 비교.

### Fig. 4 — Bayesian ML Diagnostics (M4)
> **[fig4_ml_diagnostics.png]**  
> (a) Bayesian 역추론 모델의 예측 정확도 (predicted probability vs actual label). (b) Physics-informed feature importance 분석으로, 어떤 검사 신호가 진단에 가장 기여하는지 보여줍니다.

### Fig. 5 — Process Attribution (M5)
> **[fig5_attribution.png]**  
> (a) Overlay 분산 분해 pie chart: Scanner, Mask (Pristine), Mask (Degradation), Process 기여도. (b) Bayesian changepoint detection에 의한 degradation onset 자동 감지.

### Fig. 6 — Material Comparison (5종)
> **[fig6_material_comparison.png]**  
> ULE 7973, Zerodur, Clearceram-Z, AGC AZ, Shin-Etsu Quartz 5종 소재의 fracture toughness, SCG exponent, Young's modulus, thermal conductivity, activation energy 비교 레이더 차트.

---

<!-- v2 추가: References 섹션 신규 -->
## References

[1] ASML, "EXE:5200 High-NA EUV Lithography System Technical Specifications," 2025.

[2] C. R. Kurkjian, P. K. Gupta, R. K. Brow, "The Strength of Silicate Glasses: What Do We Know, What Do We Need to Know?" *Int. J. Appl. Glass Sci.*, vol. 1, no. 3, pp. 27–37, 2010.

[3] S. M. Wiederhorn, "Subcritical Crack Growth in Ceramics," in *Fracture Mechanics of Ceramics*, vol. 2, pp. 613–646, Springer, 1974.

[4] Corning Incorporated, "ULE® Corning Code 7973 Low Thermal Expansion Glass — Product Data Sheet," 2024.

[5] SCHOTT AG, "ZERODUR® — Zero Expansion Glass Ceramic," Technical Information TIE-36, 2023.

[6] A. Karma, D. A. Kessler, H. Levine, "Phase-Field Model of Mode III Dynamic Fracture," *Phys. Rev. Lett.*, vol. 87, no. 4, 045501, 2001.

<!-- v2 신규 추가 참고문헌 -->
[7] I. R. Bolton-Lum, R. C. Dennis, P. K. Morse, E. I. Corwin, "The Ideal Glass and the Ideal Disk Packing in Two Dimensions," arXiv:2404.07492v2, Dec 2025. DOI: [10.1103/vldy-r77w](https://doi.org/10.1103/vldy-r77w). — *Polydisperse 디스크의 transient degrees of freedom으로 2D ideal glass 구현: S_config=0, 결정급 mechanical properties, hyperuniform 구조 확인.*

---

## 문의

성균관대학교 SPMDL  
권석준 교수  
📧 [이메일]  
📞 [전화번호]

---

*본 제안서의 시뮬레이션 결과는 공개 문헌 기반의 물성치를 사용하였으며, Corning 독점 데이터는 포함되어 있지 않습니다. 실제 프로젝트 수행 시 Corning으로부터 제공받는 실험 데이터를 통해 모델 calibration을 진행할 예정입니다.*
