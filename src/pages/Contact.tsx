import React, { useState } from 'react';
import { Mail, Phone, MapPin, Send, MessageCircle, Users, Clock } from 'lucide-react';
import { motion } from 'framer-motion';

const Contact: React.FC = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>): void => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e: React.FormEvent): void => {
    e.preventDefault();
    // Handle form submission
    console.log('Form submitted:', formData);
    // Reset form
    setFormData({ name: '', email: '', subject: '', message: '' });
  };

  const contactInfo = [
    {
      icon: Mail,
      title: 'Email Us',
      info: 'sumesh13055@gmail.com',
      description: 'Get in touch for support or questions'
    },
    {
      icon: Phone,
      title: 'Call Us',
      info: '+91 72880 81868',
      description: 'Mon-Fri 9AM-6PM IST'
    },
    {
      icon: MapPin,
      title: 'Visit Us',
      info: 'Hyderabad, Telangana, India',
      description: 'Traditional Art Heritage Center'
    }
  ];

  const features = [
    {
      icon: MessageCircle,
      title: 'Quick Response',
      description: 'We respond to all inquiries within 24 hours'
    },
    {
      icon: Users,
      title: 'Expert Support',
      description: 'Our team includes kolam art experts and technologists'
    },
    {
      icon: Clock,
      title: 'Always Available',
      description: 'Online support available round the clock'
    }
  ];

  return (
    <div className="min-h-screen py-12 bg-gradient-to-br from-cream to-yellow-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h1 className="font-serif text-4xl md:text-5xl font-bold text-primary-red mb-4">
            Contact Us
          </h1>
          <p className="font-sans text-xl text-gray-600 max-w-2xl mx-auto">
            Have questions about kolam patterns or our recognition system? We&apos;d love to hear from you!
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-16">
          {/* Contact Information */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="lg:col-span-1"
          >
            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8">
              <h2 className="font-serif text-2xl font-bold text-primary-red mb-8">
                Get in Touch
              </h2>
              
              <div className="space-y-6">
                {contactInfo.map((item, index) => (
                  <div key={index} className="flex items-start space-x-4">
                    <div className="w-12 h-12 bg-primary-gold rounded-full flex items-center justify-center flex-shrink-0">
                      <item.icon className="h-6 w-6 text-primary-red" />
                    </div>
                    <div>
                      <h3 className="font-sans font-semibold text-gray-900 mb-1">
                        {item.title}
                      </h3>
                      <p className="font-sans text-primary-red font-medium mb-1">
                        {item.info}
                      </p>
                      <p className="font-sans text-sm text-gray-600">
                        {item.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-8 pt-8 border-t border-gray-200">
                <h3 className="font-serif text-lg font-bold text-primary-red mb-4">
                  Why Contact Us?
                </h3>
                <div className="space-y-4">
                  {features.map((feature, index) => (
                    <div key={index} className="flex items-center space-x-3">
                      <feature.icon className="h-5 w-5 text-accent-green flex-shrink-0" />
                      <div>
                        <p className="font-sans font-medium text-gray-900 text-sm">
                          {feature.title}
                        </p>
                        <p className="font-sans text-xs text-gray-600">
                          {feature.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>

          {/* Contact Form */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="lg:col-span-2"
          >
            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8">
              <h2 className="font-serif text-2xl font-bold text-primary-red mb-8">
                Send us a Message
              </h2>
              
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label htmlFor="name" className="block font-sans font-medium text-gray-700 mb-2">
                      Full Name *
                    </label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      value={formData.name}
                      onChange={handleChange}
                      required
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-sans"
                      placeholder="Your full name"
                    />
                  </div>
                  
                  <div>
                    <label htmlFor="email" className="block font-sans font-medium text-gray-700 mb-2">
                      Email Address *
                    </label>
                    <input
                      type="email"
                      id="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      required
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-sans"
                      placeholder="your.email@example.com"
                    />
                  </div>
                </div>

                <div>
                  <label htmlFor="subject" className="block font-sans font-medium text-gray-700 mb-2">
                    Subject *
                  </label>
                  <select
                    id="subject"
                    name="subject"
                    value={formData.subject}
                    onChange={handleChange}
                    required
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-sans"
                  >
                    <option value="">Select a subject</option>
                    <option value="technical-support">Technical Support</option>
                    <option value="pattern-submission">Pattern Submission</option>
                    <option value="collaboration">Collaboration</option>
                    <option value="feedback">Feedback</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="message" className="block font-sans font-medium text-gray-700 mb-2">
                    Message *
                  </label>
                  <textarea
                    id="message"
                    name="message"
                    value={formData.message}
                    onChange={handleChange}
                    required
                    rows={6}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-sans resize-vertical"
                    placeholder="Tell us how we can help you..."
                  />
                </div>

                <div className="bg-cream rounded-lg p-4">
                  <p className="font-sans text-sm text-gray-600">
                    <strong>Note:</strong> We respect your privacy and will never share your personal information. 
                    All communications are handled confidentially according to our privacy policy.
                  </p>
                </div>

                <button
                  type="submit"
                  className="w-full bg-primary-red hover:bg-red-700 text-white font-sans font-bold py-4 px-8 rounded-lg transition-colors flex items-center justify-center"
                >
                  <Send className="h-5 w-5 mr-2" />
                  Send Message
                </button>
              </form>
            </div>
          </motion.div>
        </div>

        {/* FAQ Section */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8">
            <h2 className="font-serif text-3xl font-bold text-primary-red text-center mb-8">
              Frequently Asked Questions
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-6">
                <div>
                  <h3 className="font-sans font-semibold text-gray-900 mb-2">
                    How accurate is the pattern recognition?
                  </h3>
                  <p className="font-sans text-gray-600 text-sm leading-relaxed">
                    Our AI system achieves over 90% accuracy in recognizing traditional kolam patterns, 
                    with continuous improvements through machine learning.
                  </p>
                </div>
                
                <div>
                  <h3 className="font-sans font-semibold text-gray-900 mb-2">
                    Can I contribute my own patterns?
                  </h3>
                  <p className="font-sans text-gray-600 text-sm leading-relaxed">
                    Yes! We welcome contributions from artists and enthusiasts. 
                    Contact us to learn about our community contribution program.
                  </p>
                </div>
              </div>
              
              <div className="space-y-6">
                <div>
                  <h3 className="font-sans font-semibold text-gray-900 mb-2">
                    Is the service free to use?
                  </h3>
                  <p className="font-sans text-gray-600 text-sm leading-relaxed">
                    Basic pattern recognition and download features are completely free. 
                    Premium features for educators and commercial use are available.
                  </p>
                </div>
                
                <div>
                  <h3 className="font-sans font-semibold text-gray-900 mb-2">
                    Do you offer educational workshops?
                  </h3>
                  <p className="font-sans text-gray-600 text-sm leading-relaxed">
                    We conduct workshops for schools, cultural centers, and art institutions. 
                    Contact us to schedule a workshop in your area.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </motion.section>
      </div>
    </div>
  );
};

export default Contact;
