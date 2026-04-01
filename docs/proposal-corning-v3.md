# Glass Core/Interposer Micro-Crack 예측 관리 시뮬레이터 개발

**제안 기관:** 성균관대학교 SPMDL (권석준 교수)  
**협력 대상:** 코닝 코리아 천안 연구소  
**제안 일자:** 2026년 3월

---

## 1. 연구 배경

### Glass Core/Interposer 시장 급부상
2024년 Intel의 glass substrate 공식 발표를 시작으로 Samsung, TSMC 등 주요 파운드리가 차세대 AI 패키징 핵심 기술로 glass core/interposer 도입을 가속화하고 있음. Intel은 2026년 첫 상용 제품에서 EMIB-embedded glass core에서 "SeWaRe(micro-cracking) 문제 없음"을 강조하며 기술 성숙도를 과시했으나, 실제 양산에서는 여전히 micro-crack이 핵심 기술 장벽으로 남아 있음.

### Corning 천안의 기술적 과제
1. **TGV(Through Glass Via) 가공 시 micro-crack** — 레이저 드릴링 공정에서 열충격으로 인한 radial crack 발생
2. **CTE mismatch 기인 thermal cycling crack** — glass core(~4.8 ppm/°C)와 Cu RDL(~17 ppm/°C) 간 열팽창 차이로 인한 계면 응력 집중
3. **High throughput inspection 병목** — 기존 SAM, 광학 검사로는 양산 라인 속도 대응 한계

---

## 2. 연구 목표

천안 연구소가 보유한 풍부한 공정 데이터와 장비 인프라에, SPMDL의 물리 모델링·AI 역량을 결합하여, 천안 단독으로는 도달하기 어려운 crack 예측·진단 시스템을 공동 개발하는 것을 목표로 함.

**핵심 목표:** Glass core/interposer 양산 공정에서 micro-crack 발생 mechanism 규명 및 고속 진단 시스템 개발

**세부 목표:**
- TGV 레이저 가공 조건 최적화를 위한 crack nucleation 예측 모델 개발
- Thermal cycling 시 CTE mismatch crack growth 시뮬레이션 엔진 구축  
- 인라인 검사 데이터 기반 실시간 crack 상태 진단 AI 시스템 구현
- 천안 보유 장비 활용 가능한 실증 프로토타입 완성

---

## 3. 기술 및 방법론

본 시뮬레이터는 기존 EUV photomask 연구에서 검증된 5모듈 구조를 glass core/interposer 응용으로 완전 재설계함.

| 모듈 | Glass Core/Interposer 적용 | 천안 연구소 연계 |
|------|---------------------------|------------------|
| **M1: Nucleation** | TGV 레이저 가공 시 defect 분포 및 crack nucleation 확률 | 천안 보유 femtosecond laser 가공 데이터 활용 |
| **M2: Propagation** | CTE mismatch 기인 thermal cycling crack growth | 천안 thermal cycling chamber 실험 데이터 연동 |
| **M3: Inspection Forward** | SAM/광학/전기적 인라인 검사 신호 시뮬레이션 | 천안 C-SAM, 광학 현미경 장비 연계 |
| **M4: ML Diagnostics** | 검사 신호로부터 crack 크기/위치 역추론 | 천안 수집 검사 데이터로 ML 모델 학습 |
| **M5: Process Attribution** | TGV/RDL/PKG 공정별 crack 기여도 분해 | 천안 공정 조건 DB 기반 최적화 |

### 핵심 차별화 기술

**1. TGV Crack Physics Model**
- Femtosecond/picosecond laser 가공 시 HAZ(Heat Affected Zone) 내 잔류응력 분포 계산
- Glass thermal shock parameter와 레이저 pulse energy 상관관계 모델링

**2. CTE Mismatch Stress Analysis**  
- Glass core - Cu RDL - molding compound 3층 구조 thermal cycling FEM
- Interface delamination과 through-crack transition 임계조건 도출

**3. High-Speed Inspection Integration**
- C-SAM + 광학 검사 fusion으로 throughput 3–5배 향상
- Edge AI 기반 실시간 pass/fail 판정

---

## 4. 기대 효과

### 기술적 효과
- **TGV crack 불량률 감소** — 레이저 조건 최적화를 통한 crack density 저감 (목표: 20–30%)
- **Thermal reliability 향상** — CTE mismatch 응력 예측 기반 구조 설계 개선 (목표: 15–25%)
- **검사 throughput 향상** — 인라인 AI 진단 도입으로 검사 속도 3–5배 개선

### 사업적 효과  
- **천안 연구소 glass packaging 역량 강화** — Intel/Samsung 대상 기술 differentiation 자료 확보
- **양산 라인 조기 안정화** — crack 예측 기반 공정 조건 사전 최적화
- **차세대 glass core 개발 가속화** — ultra-low CTE glass 개발 방향 가이드라인 제시

---

## 5. 연구 수행 계획

### Phase 1: 시뮬레이터 Glass Core 적응화 (6개월)
- M1-M2 모듈 TGV/thermal cycling 특화 재설계
- 천안 레이저 가공 조건 DB와 시뮬레이션 연계
- Borofloat/AGC 등 기존 glass 대비 벤치마킹

### Phase 2: AI 진단 시스템 구축 (8개월)  
- 천안 C-SAM/광학 검사 데이터 수집 및 라벨링
- M4 ML 모델 천안 장비 특성에 맞춰 재훈련
- 실시간 진단 프로토타입 개발

### Phase 3: 현장 실증 및 기술 이전 (4개월)
- 천안 파일럿 라인에서 실증 테스트
- 공정 엔지니어 대상 시뮬레이터 교육
- 기술 문서화 및 소프트웨어 패키지 완성

### 연구진 및 협력 체계
- **SPMDL:** 시뮬레이션 엔진, ML 모델 개발
- **천안 연구소:** 실험 데이터 제공, 장비 접근, 현장 실증
- **월례 리뷰:** 천안 연구소와 진행상황 점검 및 방향 조정

---

## 6. 기존 연구 성과 활용

현재 구축된 glass crack 시뮬레이터 기반 기술을 glass core/interposer 응용으로 전환하여 개발 리스크를 최소화하면서 빠른 성과 창출 가능.

### 활용 가능한 기존 결과
- **5모듈 통합 아키텍처** — 전체 구조 재사용으로 개발 기간 단축
- **Glass crack physics 모델** — ULE 기반 모델을 borosilicate glass로 확장 적용
- **ML 진단 프레임워크** — Bayesian GP 기반 역추론 엔진 재활용

### Glass Core 특화 신규 개발
- **TGV 특화 nucleation 모델** — 레이저-glass interaction physics 추가
- **Multi-layer thermal stress 해석** — glass/Cu/mold compound 적층 구조 FEM  
- **High-speed inspection 신호 모델** — SAM/광학 fusion 알고리즘

---

## 7. 결론

Glass core/interposer는 차세대 AI 패키징의 핵심 기술이나, micro-crack 문제로 인한 양산성 확보가 업계 최대 과제임. 본 과제를 통해 천안 연구소는 crack mechanism 규명 및 고속 진단 기술을 확보하여 Intel/Samsung 등 고객사 대상 기술 우위를 선점할 수 있음.

특히 천안 보유 장비와 데이터를 최대한 활용하여 현실적이고 실용적인 솔루션 개발에 집중하며, 단기간 내 가시적 성과 창출을 목표로 함.

---

**문의:** 성균관대학교 SPMDL  
**협력 대상:** 코닝 코리아 천안 연구소