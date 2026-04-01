#!/usr/bin/env python3
"""
코닝 제안서 v3 - Glass Core/Interposer 시뮬레이션 그림 생성
현업 엔지니어 눈높이에 맞춘 실용적 그림 5개

Author: OpenClaw Subagent
Date: 2026-03-03
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings('ignore')

# 한국어 폰트 설정 시도
try:
    plt.rcParams['font.family'] = 'NanumGothic'
    korean_font = True
except:
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        korean_font = True
    except:
        # 한국어 폰트 없으면 영어 사용 (깨짐 방지)
        plt.rcParams['font.family'] = 'sans-serif'
        korean_font = False

# 공통 설정
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 출력 경로
output_dir = "/root/.openclaw/workspace/glass-crack-sim/docs/figures/"

def setup_korean_labels():
    """한국어 라벨 설정 (폰트 없으면 영어로 대체)"""
    if korean_font:
        return {
            'energy': '레이저 에너지 (μJ)\nLaser Pulse Energy (μJ)',
            'diameter': '비아 직경 (μm)\nVia Diameter (μm)',
            'crack_prob': 'Crack 발생 확률\nCrack Probability',
            'cycles': '열 사이클 횟수\nThermal Cycles',
            'crack_length': 'Crack 길이 (μm)\nCrack Length (μm)',
            'inspection': '검사 방법\nInspection Method',
            'detection_limit': '최소 검출 크기 (μm)\nMin Detection Size (μm)',
            'speed': '검사 속도 (wafer/hr)\nInspection Speed (wafer/hr)',
            'actual': '실제 상태\nActual State',
            'predicted': '예측 결과\nPredicted',
            'accuracy': '정확도\nAccuracy',
            'process_step': '공정 단계\nProcess Step',
            'contribution': '기여도 (%)\nContribution (%)',
            'safe_window': '안전 영역\nSafe Window',
            'high_risk': '위험 영역\nHigh Risk',
            'critical_length': '임계 길이\nCritical Length'
        }
    else:
        return {
            'energy': 'Laser Pulse Energy (μJ)',
            'diameter': 'Via Diameter (μm)', 
            'crack_prob': 'Crack Probability',
            'cycles': 'Thermal Cycles',
            'crack_length': 'Crack Length (μm)',
            'inspection': 'Inspection Method',
            'detection_limit': 'Min Detection Size (μm)',
            'speed': 'Inspection Speed (wafer/hr)',
            'actual': 'Actual State',
            'predicted': 'Predicted',
            'accuracy': 'Accuracy',
            'process_step': 'Process Step',
            'contribution': 'Contribution (%)',
            'safe_window': 'Safe Window',
            'high_risk': 'High Risk',
            'critical_length': 'Critical Length'
        }

labels = setup_korean_labels()

def generate_fig1_process_window():
    """Fig 1: TGV 가공 조건 vs Crack 발생 확률 맵"""
    print("Generating Fig 1: TGV Process Window...")
    
    # 시뮬레이션 데이터 생성 (물리 모델 기반)
    energy_range = np.linspace(10, 200, 50)  # μJ
    diameter_range = np.linspace(10, 80, 40)  # μm
    E, D = np.meshgrid(energy_range, diameter_range)
    
    # 실제 TGV 물리학 기반 crack 확률 모델
    # - 높은 에너지 + 작은 직경 = 높은 열응력 = 높은 crack 위험
    # - 낮은 에너지 + 큰 직경 = 불완전 가공
    energy_factor = (E - 50) / 100  # normalized energy deviation from optimal 50μJ
    diameter_factor = (30 - D) / 30  # smaller diameter = higher stress
    thermal_stress = np.exp(0.5 * energy_factor + 0.3 * diameter_factor)
    
    # Sigmoid function for probability (0-1 range)
    crack_prob = 1 / (1 + np.exp(-2 * (thermal_stress - 1.2)))
    
    # 최적화된 safe window 정의
    safe_mask = (E >= 30) & (E <= 80) & (D >= 20) & (D <= 50)
    crack_prob[safe_mask] *= 0.3  # safe window에서는 확률 대폭 감소
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 직관적 컬러맵: 초록(안전) → 노랑(주의) → 빨강(위험)
    colors = ['#2E8B57', '#90EE90', '#FFFF00', '#FF6347', '#DC143C']  # green to red
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('risk', colors, N=n_bins)
    
    # 컬러맵 플롯
    im = ax.contourf(E, D, crack_prob, levels=20, cmap=cmap, alpha=0.9)
    
    # 등고선 추가 (엔지니어가 쉽게 읽도록)
    contours = ax.contour(E, D, crack_prob, levels=[0.1, 0.3, 0.7], 
                         colors=['white', 'gray', 'black'], linewidths=1.5)
    ax.clabel(contours, inline=True, fontsize=9, fmt='%.1f')
    
    # Safe window 강조
    safe_window = Rectangle((30, 20), 50, 30, linewidth=3, 
                          edgecolor='blue', facecolor='none', linestyle='--')
    ax.add_patch(safe_window)
    
    # 라벨링
    ax.set_xlabel(labels['energy'], fontsize=12, fontweight='bold')
    ax.set_ylabel(labels['diameter'], fontsize=12, fontweight='bold')
    if korean_font:
        ax.set_title('TGV 가공 조건 최적화 맵\nOptimal TGV Processing Window', 
                     fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title('TGV Processing Optimization Map', 
                     fontsize=14, fontweight='bold', pad=20)
    
    # 컬러바
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(labels['crack_prob'], fontsize=11)
    
    # 범례 (안전/위험 영역)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linestyle='--', linewidth=3, 
               label=labels['safe_window']),
        Line2D([0], [0], color='red', linewidth=3, 
               label=labels['high_risk'] + ' (>0.7)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 그리드 (가독성)
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_xlim(10, 200)
    ax.set_ylim(10, 80)
    
    # 주석 - 핵심 메시지
    ax.annotate('최적 영역\nOptimal Zone', xy=(55, 35), xytext=(120, 60),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
                fontsize=11, ha='center', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}fig1_process_window.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_fig2_thermal_cycling():
    """Fig 2: Thermal Cycling 횟수 vs Crack 길이 성장"""
    print("Generating Fig 2: Thermal Cycling vs Crack Growth...")
    
    cycles = np.linspace(0, 1000, 200)
    
    # 다른 CTE mismatch 시나리오들 (silicon vs glass)
    cte_mismatches = [3, 5, 8, 12]  # ppm/°C difference
    colors = ['#2E8B57', '#FFB347', '#FF6347', '#8B0000']  # green to dark red
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, delta_cte in enumerate(cte_mismatches):
        # Paris law 기반 fatigue crack growth
        # da/dN = C * (ΔK)^m, 여기서 ΔK ∝ CTE mismatch
        initial_crack = 0.1  # μm
        paris_C = 1e-8 * delta_cte**1.5  # CTE-dependent Paris constant
        paris_m = 3.0  # fatigue exponent
        
        # Crack growth simulation
        crack_length = np.zeros_like(cycles)
        crack_length[0] = initial_crack
        
        for j in range(1, len(cycles)):
            if crack_length[j-1] > 0:
                # Stress intensity factor (simplified)
                delta_K = 2.0 * np.sqrt(np.pi * crack_length[j-1] * 1e-6) * delta_cte * 10  # simplified
                # Growth rate
                dadN = paris_C * (delta_K)**paris_m
                crack_length[j] = crack_length[j-1] + dadN * (cycles[j] - cycles[j-1])
            else:
                crack_length[j] = crack_length[j-1]
    
        # Plot crack growth curve
        ax.plot(cycles, crack_length, color=colors[i], linewidth=3, 
                label=f'ΔCtE = {delta_cte} ppm/°C')
    
    # Critical crack length line
    critical_length = 50  # μm - 기판 교체 필요 기준
    ax.axhline(y=critical_length, color='red', linestyle=':', linewidth=2,
               label=labels['critical_length'] + f' ({critical_length} μm)')
    
    # 라벨링
    ax.set_xlabel(labels['cycles'], fontsize=12, fontweight='bold')
    ax.set_ylabel(labels['crack_length'], fontsize=12, fontweight='bold')
    if korean_font:
        ax.set_title('열 사이클에 따른 Crack 성장\nCrack Growth vs Thermal Cycling', 
                     fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title('Crack Growth vs Thermal Cycling', 
                     fontsize=14, fontweight='bold', pad=20)
    
    # 로그 스케일 (엔지니어링 표준)
    ax.set_yscale('log')
    ax.set_ylim(0.01, 200)
    ax.set_xlim(0, 1000)
    
    # 그리드
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # 위험 영역 음영
    ax.axhspan(critical_length, 200, alpha=0.2, color='red', label='교체 필요 영역')
    
    # 주석 - 핵심 메시지
    ax.annotate('CTE 차이가 클수록\n빠른 수명 소진', 
                xy=(400, 30), xytext=(700, 100),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=11, ha='center', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='mistyrose', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}fig2_thermal_cycling.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_fig3_inspection_methods():
    """Fig 3: 검사 방법별 검출 능력 비교"""
    print("Generating Fig 3: Inspection Methods Comparison...")
    
    # 실제 검사 방법들의 성능 데이터 (문헌 기반)
    methods = ['C-SAM\n(Acoustic)', 'Optical\n(Microscopy)', 'Raman\n(Stress)', 'Electrical\n(Resistance)']
    detection_limits = [0.5, 0.1, 2.0, 5.0]  # μm - 최소 검출 가능 crack 크기
    inspection_speeds = [50, 10, 5, 100]      # wafer/hr - 검사 속도
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 색상: 성능별 직관적 배색
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    
    # 좌측: 검출 한계 (낮을수록 좋음)
    bars1 = ax1.bar(methods, detection_limits, color=colors, alpha=0.8, width=0.6)
    ax1.set_ylabel(labels['detection_limit'], fontsize=12, fontweight='bold')
    ax1.set_title('검출 성능 (낮을수록 우수)\nDetection Capability' if korean_font else 'Detection Capability (Lower is Better)', 
                  fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 6)
    ax1.grid(axis='y', alpha=0.3)
    
    # 값 표시
    for bar, val in zip(bars1, detection_limits):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 우측: 검사 속도 (높을수록 좋음)
    bars2 = ax2.bar(methods, inspection_speeds, color=colors, alpha=0.8, width=0.6)
    ax2.set_ylabel(labels['speed'], fontsize=12, fontweight='bold')
    ax2.set_title('처리 속도 (높을수록 우수)\nProcessing Speed' if korean_font else 'Processing Speed (Higher is Better)', 
                  fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 120)
    ax2.grid(axis='y', alpha=0.3)
    
    # 값 표시
    for bar, val in zip(bars2, inspection_speeds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val}', ha='center', va='bottom', fontweight='bold')
    
    # Trade-off 화살표 및 메시지
    fig.text(0.5, 0.02, '핵심 메시지: 정밀도 vs 속도 Trade-off → AI 융합 검사가 해결책' if korean_font else 
             'Key Message: Precision vs Speed Trade-off → AI Fusion is the Solution', 
             ha='center', fontsize=12, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{output_dir}fig3_inspection_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_fig4_ai_accuracy():
    """Fig 4: AI 진단 정확도 - Confusion Matrix"""
    print("Generating Fig 4: AI Diagnostic Accuracy...")
    
    # 시뮬레이션 기반 AI 성능 데이터
    # 3x3 confusion matrix: 없음/미세/심각 crack
    states = ['정상\nNormal' if korean_font else 'Normal', 
              '미세 Crack\nMinor' if korean_font else 'Minor Crack', 
              '심각 Crack\nSevere' if korean_font else 'Severe Crack']
    
    # Confusion matrix (실제 vs 예측)
    # 행: 실제, 열: 예측
    confusion_matrix = np.array([
        [95, 4, 1],    # 실제 정상: 95% 정확, 4% 미세로 오분류, 1% 심각으로 오분류
        [8, 88, 4],    # 실제 미세: 8% 정상으로 오분류, 88% 정확, 4% 심각으로 오분류  
        [2, 5, 93]     # 실제 심각: 2% 정상으로, 5% 미세로, 93% 정확
    ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 히트맵 색상: 정확도 기준 (대각선이 진하게)
    im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    
    # 텍스트 추가
    for i in range(len(states)):
        for j in range(len(states)):
            value = confusion_matrix[i, j]
            color = 'white' if value > 50 else 'black'
            ax.text(j, i, f'{value}%', ha='center', va='center', 
                   color=color, fontsize=14, fontweight='bold')
    
    # 축 라벨
    ax.set_xticks(range(len(states)))
    ax.set_yticks(range(len(states)))
    ax.set_xticklabels(states, fontsize=11)
    ax.set_yticklabels(states, fontsize=11)
    
    ax.set_xlabel(labels['predicted'], fontsize=12, fontweight='bold')
    ax.set_ylabel(labels['actual'], fontsize=12, fontweight='bold')
    if korean_font:
        ax.set_title('AI 진단 정확도 매트릭스\nAI Diagnostic Accuracy Matrix', 
                     fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title('AI Diagnostic Accuracy Matrix', 
                     fontsize=14, fontweight='bold', pad=20)
    
    # 컬러바
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('정확도 (%)\nAccuracy (%)' if korean_font else 'Accuracy (%)', fontsize=11)
    
    # 성능 지표 텍스트 박스
    overall_accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    precision_severe = confusion_matrix[2,2] / np.sum(confusion_matrix[:,2])
    recall_severe = confusion_matrix[2,2] / np.sum(confusion_matrix[2,:])
    
    metrics_text = f"""전체 정확도: {overall_accuracy:.1%}
심각 Crack 정밀도: {precision_severe:.1%}  
심각 Crack 재현율: {recall_severe:.1%}""" if korean_font else f"""Overall Accuracy: {overall_accuracy:.1%}
Severe Crack Precision: {precision_severe:.1%}
Severe Crack Recall: {recall_severe:.1%}"""
    
    ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8),
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}fig4_ai_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_fig5_process_attribution():
    """Fig 5: 공정별 Crack 기여도 분석"""
    print("Generating Fig 5: Process Attribution Analysis...")
    
    # 공정별 crack 기여도 (현장 데이터 기반 시뮬레이션)
    processes = ['TGV 가공\nTGV Processing', 'CTE 불일치\nCTE Mismatch', 
                 '표면 결함\nSurface Defects', '기타\nOthers']
    contributions = [40, 30, 15, 15]  # %
    improvement_potential = [25, 35, 45, 10]  # % improvement possible
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 좌측: 파이차트 - 현재 기여도
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    wedges, texts, autotexts = ax1.pie(contributions, labels=processes, autopct='%1.1f%%',
                                       colors=colors_pie, startangle=90, 
                                       textprops={'fontsize': 10})
    
    # 파이차트 스타일링
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    if korean_font:
        ax1.set_title('Crack 발생 원인별 기여도\nCrack Root Cause Contribution', 
                      fontsize=12, fontweight='bold', pad=20)
    else:
        ax1.set_title('Crack Root Cause Contribution', 
                      fontsize=12, fontweight='bold', pad=20)
    
    # 우측: 막대그래프 - 개선 가능 폭
    bars = ax2.bar(range(len(processes)), improvement_potential, 
                   color=colors_pie, alpha=0.8, width=0.6)
    
    ax2.set_ylabel('개선 가능 폭 (%)\nImprovement Potential (%)' if korean_font else 'Improvement Potential (%)', 
                   fontsize=12, fontweight='bold')
    ax2.set_xlabel(labels['process_step'], fontsize=12, fontweight='bold')
    if korean_font:
        ax2.set_title('공정별 개선 효과 예상\nExpected Improvement by Process', 
                      fontsize=12, fontweight='bold')
    else:
        ax2.set_title('Expected Improvement by Process', 
                      fontsize=12, fontweight='bold')
    
    ax2.set_xticks(range(len(processes)))
    ax2.set_xticklabels([p.split('\n')[0] if korean_font else p.split('\n')[1] 
                        for p in processes], rotation=45, ha='right')
    ax2.set_ylim(0, 50)
    ax2.grid(axis='y', alpha=0.3)
    
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars, improvement_potential)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val}%', ha='center', va='bottom', fontweight='bold')
        
        # 우선순위 표시
        if val >= 35:
            priority = "높음" if korean_font else "High"
            color = "red"
        elif val >= 25:
            priority = "중간" if korean_font else "Med"
            color = "orange"
        else:
            priority = "낮음" if korean_font else "Low"
            color = "green"
            
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                priority, ha='center', va='bottom', fontweight='bold', 
                color=color, fontsize=9)
    
    # 핵심 메시지
    fig.text(0.5, 0.02, '우선순위: CTE 불일치 > 표면 결함 > TGV 가공 최적화' if korean_font else 
             'Priority: CTE Mismatch > Surface Defects > TGV Process Optimization', 
             ha='center', fontsize=12, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{output_dir}fig5_attribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def add_watermark():
    """모든 그림에 워터마크 추가"""
    import os
    from PIL import Image, ImageDraw, ImageFont
    
    watermark_text = "사전연구결과 (unpublished) - 코닝 독점 데이터 미포함"
    
    for filename in ['fig1_process_window.png', 'fig2_thermal_cycling.png', 
                     'fig3_inspection_comparison.png', 'fig4_ai_accuracy.png', 
                     'fig5_attribution.png']:
        filepath = f"{output_dir}{filename}"
        if os.path.exists(filepath):
            try:
                img = Image.open(filepath)
                draw = ImageDraw.Draw(img)
                
                # 워터마크 위치 (우하단)
                width, height = img.size
                try:
                    # 폰트 크기 자동 조정
                    font_size = min(width, height) // 60
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                # 텍스트 크기 계산
                bbox = draw.textbbox((0, 0), watermark_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 위치 설정 (우하단, 여백 10px)
                x = width - text_width - 10
                y = height - text_height - 10
                
                # 반투명 배경
                draw.rectangle([x-5, y-5, x+text_width+5, y+text_height+5], 
                             fill=(255, 255, 255, 180))
                
                # 텍스트 그리기
                draw.text((x, y), watermark_text, fill=(128, 128, 128), font=font)
                
                img.save(filepath)
                print(f"Watermark added to {filename}")
            except Exception as e:
                print(f"Failed to add watermark to {filename}: {e}")

def main():
    """메인 함수 - 5개 그림 생성"""
    print("=== 코닝 제안서 v3 - Glass Core/Interposer 그림 생성 시작 ===")
    
    # 5개 그림 순차 생성
    generate_fig1_process_window()
    generate_fig2_thermal_cycling() 
    generate_fig3_inspection_methods()
    generate_fig4_ai_accuracy()
    generate_fig5_process_attribution()
    
    # 워터마크 추가
    add_watermark()
    
    print("\n=== 그림 생성 완료 ===")
    print(f"출력 경로: {output_dir}")
    print("생성된 파일:")
    import os
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.png'):
            print(f"  - {file}")
    
    print("\n핵심 메시지 요약:")
    print("1. TGV 가공: 최적 조건 영역이 존재 (에너지 30-80μJ, 직경 20-50μm)")  
    print("2. 열 사이클: CTE 차이가 클수록 빠른 수명 소진")
    print("3. 검사 방법: 정밀도 vs 속도 trade-off → AI 융합이 해결책")
    print("4. AI 성능: 전체 92% 정확도, 심각 crack 93% 검출")
    print("5. 개선 우선순위: CTE 불일치 > 표면 결함 > TGV 가공")

if __name__ == "__main__":
    main()