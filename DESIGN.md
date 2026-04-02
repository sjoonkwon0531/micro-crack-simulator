# Design System: Glass Micro-Crack Simulator (Corning Edition)
# Light Mode Design System for Scientific/Engineering Simulator

## 1. Visual Theme & Atmosphere
Light-mode-first scientific dashboard. Clean white canvas where data visualizations emerge with clarity and professionalism.
Apple-like design precision suitable for Corning executive demo and extended daily use.
Trust-building through clean data presentation and honest disclaimers.

## 2. Color Palette & Roles

### Background Surfaces
- **Canvas White** (`#ffffff`): Main background
- **Panel Light** (`#f7f8fa`): Sidebar, panel backgrounds
- **Surface** (`#ffffff`): Card backgrounds, elevated areas
- **Surface Hover** (`#f0f2f5`): Hover states

### Text & Content
- **Primary Text** (`#1a1a2e`): Main text (near-black, NOT pure black)
- **Secondary Text** (`#4a5568`): Descriptions, body text
- **Tertiary Text** (`#8a8f98`): Metadata, placeholders
- **Muted Text** (`#62666d`): Timestamps, disabled

### Brand Accent (Corning Blue)
- **Primary Blue** (`#0066B1`): Corning corporate blue — CTAs, key metrics
- **Light Blue** (`#4A9FD9`): Interactive elements, links, active states
- **Hover Blue** (`#7BB8E3`): Hover states on accent elements
- **Dark Blue** (`#003D6B`): Headers, section backgrounds

### Status Colors
- **Success Green** (`#10b981`): Pass indicators, improvements
- **Warning Orange** (`#f59e0b`): Caution, estimates
- **Danger Red** (`#ef4444`): Critical issues, failures

### Border & Divider
- **Border Subtle** (`rgba(0,0,0,0.08)`): Default borders
- **Border Standard** (`rgba(0,0,0,0.10)`): Cards, inputs
- **Border Accent** (`rgba(0,102,177,0.3)`): Highlighted sections

## 3. Typography Rules
- **Primary Font**: Inter, -apple-system, system-ui, sans-serif
- **Monospace**: SF Mono, Menlo, monospace (for values/data)
- **Headings**: Inter 500-600 weight, negative letter-spacing at large sizes
- **Body**: Inter 400, 16px, line-height 1.5
- **Data Values**: SF Mono 500, tabular-nums for alignment
- **Metric Cards**: 1.8rem bold for values, 0.85rem for labels

## 4. Component Stylings

### Metric Cards
- Background: `#ffffff`
- Border: `1px solid rgba(0,0,0,0.08)`
- Border-radius: 12px
- Padding: 20px
- Shadow: `0 1px 3px rgba(0,0,0,0.08)`
- Hover shadow: `0 2px 6px rgba(0,0,0,0.12)`
- Success variant: left-border 3px solid #10b981
- Warning variant: left-border 3px solid #f59e0b
- Danger variant: left-border 3px solid #ef4444

### Cards & Containers
- Background: `#ffffff`
- Border: `1px solid rgba(0,0,0,0.08)`
- Border-radius: 8px (standard), 12px (featured)
- Hover: subtle shadow increase

### Buttons (Streamlit)
- Primary: background #0066B1, text white, radius 6px
- Secondary: background #f0f2f5, text #1a1a2e, radius 6px
- Hover: smooth transition 0.2s

### Tabs (Streamlit)
- Inactive: text #8a8f98, no border
- Active: text #0066B1, bottom-border 2px solid #0066B1
- Hover: text #4a5568, background #f0f2f5

### Plotly Charts
- Paper bgcolor: #ffffff
- Plot bgcolor: #ffffff
- Grid: rgba(0,0,0,0.06)
- Font color: #4a5568
- Colorway: ['#0066B1', '#4A9FD9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

### Warning/Info Boxes
- Warning: background rgba(245,158,11,0.1), border-left 3px solid #f59e0b
- Info: background rgba(0,102,177,0.08), border-left 3px solid #0066B1
- Caption disclaimers: #8a8f98, italic, 0.85rem

## 5. Layout Principles
- Spacing base unit: 8px
- Section padding: 24px-32px vertical
- Card gap: 16px
- Max content in sidebar: clean grouping with expanders
- Tab content: generous padding, clear section headers

## 6. Depth & Elevation
- Level 0 (Flat): No shadow, #ffffff bg
- Level 1 (Surface): #ffffff bg + subtle border rgba(0,0,0,0.08)
- Level 2 (Card): #ffffff bg + standard border + light shadow 0 1px 3px rgba(0,0,0,0.08)
- Level 3 (Elevated): 0 2px 6px rgba(0,0,0,0.12)
- Level 4 (Modal): multi-layer shadow stack

## 7. Do's and Don'ts
### Do
- Use Corning Blue (#0066B1) sparingly for key actions and metrics only
- Add "(Est.)" to all estimated/simulated values
- Include physics disclaimers in caption style
- Use data-driven visualizations with clear axis labels
- Keep status colors semantic (green=good, orange=caution, red=critical)
- Use near-black (#1a1a2e) for primary text, NOT pure black

### Don't
- Don't use pure black (#000000) for text — use #1a1a2e
- Don't use bright colors for decoration — reserve for data encoding
- Don't hide uncertainty — show ranges and confidence intervals
- Don't overcrowd dashboards — use tabs for progressive disclosure
- Don't use harsh shadows — keep them subtle for professional look
