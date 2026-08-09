import React, { createContext, useContext, useEffect, useState } from 'react'

const STR: Record<string, Record<string, string>> = {
  en: {
    home: 'Home', bridge: 'Live Bridge', animator: 'Animator', translate: 'Translate', about: 'About', getStarted: 'Get Started',
    heroBadge: 'Accessibility-First Communication', heroT1: 'Breaking Barriers Between', heroT2: 'Sign, Speech & Text',
    heroSub: 'CommuniBridge is a real-time, multilingual platform that connects deaf, mute and speech-impaired individuals with everyone.',
    launch: '🚀 Launch Live Bridge', tryAnim: '▶ Try Sign Animator', listening: 'Listening...', converting: '● Converting to Sign',
    featTitle: 'Everything You Need', howTitle: 'How It Works',
    f1t: 'Sign to Text & Speech', f1d: 'Real-time sign language detection and conversion to text and speech.',
    f2t: 'Speech to Sign & Text', f2d: 'Convert spoken words into sign language animations and readable text.',
    f3t: 'Sign Animator', f3d: 'AI-powered avatar animations for smooth sign language delivery.',
    f4t: 'Multi-language Support', f4d: 'Communicate across languages with smart translation and localization.',
    f5t: 'Conversation History', f5d: 'Save, review, and manage your conversations securely.',
    f6t: 'Real-time & Offline', f6d: 'Lightning-fast responses with offline mode for seamless communication.',
    s1t: 'Choose Your Input', s1d: 'Use sign language, speech, or text to start the conversation.',
    s2t: 'AI Processes Instantly', s2d: 'Our AI understands and converts your input in real-time.',
    s3t: 'Get Connected', s3d: 'Receive output as sign animation, speech, or text — instantly.',
  },
  hi: {
    home: 'होम', bridge: 'लाइव ब्रिज', animator: 'एनिमेटर', translate: 'अनुवाद', about: 'परिचय', getStarted: 'शुरू करें',
    heroBadge: 'एक्सेसिबिलिटी-फर्स्ट कम्युनिकेशन', heroT1: 'बाधाओं को तोड़ते हुए', heroT2: 'साइन, वाणी और टेक्स्ट के बीच',
    heroSub: 'CommuniBridge एक रियल-टाइम, बहुभाषी प्लेटफ़ॉर्म है जो बहरे, गूंगे और वाक्-बाधित व्यक्तियों को सभी से जोड़ता है।',
    launch: '🚀 लाइव ब्रिज शुरू करें', tryAnim: '▶ साइन एनिमेटर आज़माएँ', listening: 'सुन रहा है...', converting: '● साइन में बदल रहा है',
    featTitle: 'आपको सब कुछ मिलेगा', howTitle: 'यह कैसे काम करता है',
    f1t: 'साइन → टेक्स्ट और वाणी', f1d: 'साइन लैंग्वेज का रियल-टाइम डिटेक्शन और टेक्स्ट/वाणी में रूपांतरण।',
    f2t: 'वाणी → साइन और टेक्स्ट', f2d: 'बोले गए शब्दों को साइन एनिमेशन और पढ़ने योग्य टेक्स्ट में बदलें।',
    f3t: 'साइन एनिमेटर', f3d: 'सुचारू साइन वितरण के लिए AI-संचालित एनिमेशन।',
    f4t: 'बहुभाषा समर्थन', f4d: 'स्मार्ट अनुवाद के साथ हर भाषा में संवाद करें।',
    f5t: 'बातचीत इतिहास', f5d: 'अपनी बातचीत को सुरक्षित रूप से सहेजें और प्रबंधित करें।',
    f6t: 'रियल-टाइम और ऑफ़लाइन', f6d: 'ऑफ़लाइन मोड के साथ बिजली-सी तेज़ प्रतिक्रिया।',
    s1t: 'अपना इनपुट चुनें', s1d: 'बातचीत शुरू करने के लिए साइन, वाणी या टेक्स्ट उपयोग करें।',
    s2t: 'AI तुरंत प्रोसेस करे', s2d: 'हमारा AI आपके इनपुट को रियल-टाइम में समझता और बदलता है।',
    s3t: 'जुड़ें', s3d: 'आउटपुट पाएँ — साइन एनिमेशन, वाणी या टेक्स्ट के रूप में, तुरंत।',
  },
}

const Ctx = createContext<{ lang: string; setLang: (l: string) => void; t: (k: string) => string }>({ lang: 'en', setLang: () => {}, t: k => k })

export function LangProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLang] = useState(localStorage.getItem('lang') || 'en')
  useEffect(() => { localStorage.setItem('lang', lang) }, [lang])
  const t = (k: string) => STR[lang]?.[k] ?? STR.en[k] ?? k
  return <Ctx.Provider value={{ lang, setLang, t }}>{children}</Ctx.Provider>
}
export const useLang = () => useContext(Ctx)