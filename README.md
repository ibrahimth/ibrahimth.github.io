# Ibrahim Al-Thamary's Academic Website

[![Deploy to GitHub Pages](https://github.com/ibrahimth/ibrahimth.github.io/actions/workflows/deploy.yml/badge.svg)](https://github.com/ibrahimth/ibrahimth.github.io/actions/workflows/deploy.yml)

Personal academic website for Ibrahim Al-Thamary, Ph.D., featuring research publications, teaching materials, and interactive educational content.

## 🌐 Live Website

**Main Site**: [https://ibrahimth.github.io](https://ibrahimth.github.io)

**COE292 Interactive Studio**: [https://ibrahimth.github.io/teaching/coe-292/](https://ibrahimth.github.io/teaching/coe-292/)

## 📊 Analytics & Visitor Tracking

The website includes comprehensive visitor analytics:

- **Visitor Counter**: Real-time display of total site visitors
- **Google Analytics**: Detailed interaction tracking including:
  - Page views and session duration
  - Navigation clicks and social media interactions
  - COE292 course access and engagement
  - Tab switches and demo interactions
- **Event Tracking**: Custom events for educational content usage

### Analytics Features:
- 📈 Animated visitor counter on the main page
- 🎯 Interaction tracking for all clickable elements
- 📚 Specialized tracking for educational content
- 🔄 Real-time updates and engagement metrics

## 🏗️ Project Structure

```
ibrahimth.github.io/
├── index.html                 # Main personal website
├── teaching/
│   └── coe-292/              # COE292 Interactive Studio
│       ├── src/              # React source files
│       ├── package.json      # Dependencies
│       ├── vite.config.js    # Build configuration
│       └── dist/             # Built files (auto-generated)
├── .github/
│   └── workflows/
│       └── deploy.yml        # GitHub Actions deployment
└── README.md
```

## 🎓 COE292 - Introduction to Artificial Intelligence

Interactive learning platform featuring:

### Topic 1 - Interactive Studio
- Search algorithms demonstrations
- Interactive problem-solving exercises
- Visual algorithm comparisons

### Topic 2 - Goal Trees & Problem Solving
- 🗼 **Tower of Hanoi** - Interactive canvas demo with hints and undo
- 🌳 **Goal Tree Visualization** - AND/OR tree structure exploration
- 🔄 **Recursion Sandbox** - Factorial calculations with execution tracing
- 🔍 **Generate & Test** - Naive vs. pruned search comparisons
- 📝 **Self-Check Quiz** - Interactive assessment with randomized answers
- 🧮 **Symbolic Integration** - Problem reduction examples

## 🚀 Development & Deployment

### Prerequisites
- Node.js (v18 or higher)
- npm

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ibrahimth/ibrahimth.github.io.git
   cd ibrahimth.github.io
   ```

2. **Develop COE292 Interactive Studio**:
   ```bash
   cd teaching/coe-292
   npm install
   npm run dev
   ```

3. **Build for production**:
   ```bash
   npm run build
   ```

### Automatic Deployment

The site automatically deploys via **GitHub Actions** when changes are pushed to the `main` branch:

1. 🔨 Builds the COE292 React application
2. 📦 Combines with the main static site
3. 🚀 Deploys to GitHub Pages
4. 📊 Analytics start tracking immediately

## 🛠️ Technologies Used

### Main Website
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Styling**: Custom CSS with animations and responsive design
- **Analytics**: Google Analytics + CountAPI visitor counter

### COE292 Interactive Studio
- **Frontend**: React 18, TypeScript
- **UI Components**: Custom shadcn/ui components
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Build Tool**: Vite
- **Canvas Graphics**: HTML5 Canvas API

## 📈 Visitor Analytics Dashboard

The website tracks various metrics:

### Main Website Metrics:
- 👥 **Total Visitors**: Live counter with API integration
- 🔗 **Navigation Clicks**: Track section engagement
- 📱 **Social Media Clicks**: Track external link engagement
- 🎓 **Course Access**: Track COE292 studio visits

### COE292 Studio Metrics:
- 📑 **Tab Switches**: Track which topics are most popular
- 🎮 **Interactive Demos**: Monitor hands-on learning engagement
- 📊 **Quiz Performance**: Track learning assessment usage
- ⏱️ **Session Duration**: Understand learning patterns

## 🎯 Educational Impact

### Learning Analytics:
- **Student Engagement**: Track which AI concepts generate most interest
- **Learning Patterns**: Understand how students navigate content
- **Content Effectiveness**: Monitor demo usage and quiz completion
- **Accessibility Metrics**: Ensure content reaches all learners

## 📊 SEO & Performance

- ✅ **Semantic HTML** for accessibility
- 🚀 **Optimized Loading** with lazy loading and code splitting
- 📱 **Mobile Responsive** design
- 🔍 **SEO Optimized** with meta descriptions and structured data
- 📈 **Performance Metrics** tracking via analytics

## 🔧 Configuration

### Setting up Google Analytics:
1. Create a Google Analytics 4 property
2. Replace `G-XXXXXXXXXX` in both `index.html` files with your tracking ID
3. Configure goals and conversion tracking in GA4 dashboard

### Visitor Counter API:
- Uses CountAPI.xyz for reliable visitor counting
- Automatic fallback if service is unavailable
- Animated counter display with smooth transitions

## 📝 Content Management

### Adding New Publications:
Edit the publications section in `index.html` following the existing format.

### Updating COE292 Content:
Modify React components in `teaching/coe-292/src/` and redeploy.

### Analytics Configuration:
Update event tracking in the JavaScript sections for new interactive elements.

## 🤝 Contributing

This is an academic website for educational purposes. For suggestions or improvements:

1. 🐛 **Bug Reports**: Create an issue describing the problem
2. 💡 **Feature Requests**: Suggest improvements for educational content
3. 📚 **Content Updates**: Help improve learning materials

## 📧 Contact

**Ibrahim Al-Thamary, Ph.D.**  
Postdoctoral Fellow  
Intelligent Secure Systems IRC, KFUPM  
📧 Email: i.ibrahim42009@gmail.com  
🌐 Website: https://ibrahimth.github.io

---

🎓 **Built for AI Education Excellence** | 📊 **Analytics-Driven Learning** | 🚀 **Continuously Improving**

*Empowering the next generation of AI researchers and practitioners through interactive, data-driven education.*