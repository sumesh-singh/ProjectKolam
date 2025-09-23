import React from 'react';
import { Flower2, Users, BookOpen, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';

const About: React.FC = () => {
  const fadeInUp = {
    initial: { opacity: 0, y: 60 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 }
  };

  const traditions = [
    {
      title: "Daily Ritual",
      description: "Kolam is traditionally drawn at dawn in front of homes as a daily spiritual practice",
      icon: "🌅"
    },
    {
      title: "Festival Celebrations",
      description: "Elaborate kolam designs mark special occasions and religious festivals",
      icon: "🎉"
    },
    {
      title: "Community Bonding",
      description: "Women gather to create kolam together, strengthening community ties",
      icon: "👥"
    },
    {
      title: "Seasonal Patterns",
      description: "Different patterns reflect seasons, harvests, and natural cycles",
      icon: "🌾"
    }
  ];

  const benefits = [
    {
      title: "Meditation & Focus",
      description: "The repetitive patterns promote mindfulness and concentration",
      icon: Sparkles,
      color: "text-accent-indigo"
    },
    {
      title: "Mathematical Learning",
      description: "Kolam teaches geometry, symmetry, and mathematical concepts",
      icon: BookOpen,
      color: "text-accent-green"
    },
    {
      title: "Cultural Preservation",
      description: "Maintains traditions and passes knowledge to future generations",
      icon: Users,
      color: "text-primary-red"
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
            About Kolam Art
          </h1>
          <p className="font-sans text-xl text-gray-600 max-w-3xl mx-auto">
            Discover the rich history and cultural significance of one of South India&apos;s most beautiful traditional art forms
          </p>
        </motion.div>

        {/* Introduction */}
        <motion.section
          {...fadeInUp}
          className="mb-16"
        >
          <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8 md:p-12">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
              <div>
                <h2 className="font-serif text-3xl font-bold text-primary-red mb-6">
                  What is Kolam?
                </h2>
                <div className="space-y-4 font-sans text-gray-700 leading-relaxed">
                  <p>
                    Kolam is a traditional decorative art form from Tamil Nadu, South India, 
                    where intricate patterns are drawn on the ground using rice flour, 
                    chalk powder, or white rock powder.
                  </p>
                  <p>
                    These geometric and artistic patterns are typically created at the 
                    entrance of homes, temples, and other significant places as a daily 
                    ritual, especially during dawn hours.
                  </p>
                  <p>
                    More than just decoration, kolam represents the harmony between 
                    humans and nature, serving as a spiritual practice that brings 
                    prosperity, positive energy, and protection to the household.
                  </p>
                </div>
              </div>
              
              <div className="relative">
                <div className="bg-gradient-to-br from-primary-red to-red-700 rounded-2xl p-8 text-white">
                  <Flower2 className="h-16 w-16 text-primary-gold mb-6" />
                  <h3 className="font-serif text-2xl font-bold mb-4">Sacred Geometry</h3>
                  <p className="font-sans leading-relaxed">
                    Each kolam pattern is based on mathematical principles and sacred 
                    geometry, representing the infinite nature of creation and the 
                    interconnectedness of all life.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Types of Kolam */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <h2 className="font-serif text-3xl font-bold text-primary-red text-center mb-12">
            Types of Kolam
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8 text-center">
              <div className="w-20 h-20 bg-primary-red rounded-full flex items-center justify-center mx-auto mb-6">
                <span className="text-3xl text-white font-bold">•</span>
              </div>
              <h3 className="font-serif text-xl font-bold text-primary-red mb-4">Pulli Kolam</h3>
              <p className="font-sans text-gray-600 leading-relaxed">
                Dot-based patterns where designs are created by connecting dots in a grid. 
                These require careful planning and mathematical precision.
              </p>
            </div>

            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8 text-center">
              <div className="w-20 h-20 bg-accent-indigo rounded-full flex items-center justify-center mx-auto mb-6">
                <span className="text-3xl text-white font-bold">~</span>
              </div>
              <h3 className="font-serif text-xl font-bold text-primary-red mb-4">Sikku Kolam</h3>
              <p className="font-sans text-gray-600 leading-relaxed">
                Intricate line patterns that form continuous loops without breaks. 
                These represent the cycle of life and eternal connection.
              </p>
            </div>

            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8 text-center">
              <div className="w-20 h-20 bg-accent-green rounded-full flex items-center justify-center mx-auto mb-6">
                <span className="text-3xl text-white font-bold">✿</span>
              </div>
              <h3 className="font-serif text-xl font-bold text-primary-red mb-4">Freehand Kolam</h3>
              <p className="font-sans text-gray-600 leading-relaxed">
                Artistic designs drawn without dots or guidelines, allowing for 
                creative expression and personal interpretation.
              </p>
            </div>
          </div>
        </motion.section>

        {/* Cultural Traditions */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <h2 className="font-serif text-3xl font-bold text-primary-red text-center mb-12">
            Cultural Traditions
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {traditions.map((tradition, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white rounded-xl shadow-lg border border-gray-200 p-6 text-center hover:shadow-xl transition-shadow"
              >
                <div className="text-4xl mb-4">{tradition.icon}</div>
                <h3 className="font-serif text-lg font-bold text-primary-red mb-3">
                  {tradition.title}
                </h3>
                <p className="font-sans text-sm text-gray-600 leading-relaxed">
                  {tradition.description}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Benefits */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <h2 className="font-serif text-3xl font-bold text-primary-red text-center mb-12">
            Benefits of Kolam Practice
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {benefits.map((benefit, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: index % 2 === 0 ? -30 : 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8"
              >
                <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-6 ${
                  benefit.color === 'text-accent-indigo' ? 'bg-indigo-100' :
                  benefit.color === 'text-accent-green' ? 'bg-green-100' : 'bg-red-100'
                }`}>
                  <benefit.icon className={`h-8 w-8 ${benefit.color}`} />
                </div>
                <h3 className="font-serif text-xl font-bold text-primary-red mb-4">
                  {benefit.title}
                </h3>
                <p className="font-sans text-gray-600 leading-relaxed">
                  {benefit.description}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Call to Action */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center"
        >
          <div className="bg-gradient-to-r from-primary-red to-red-800 rounded-2xl p-12 text-white">
            <h2 className="font-serif text-3xl font-bold mb-6">
              Preserve This Beautiful Art Form
            </h2>
            <p className="font-sans text-xl text-red-100 mb-8 max-w-2xl mx-auto">
              Help us preserve and share the beauty of kolam art with the world through 
              our digital recognition and recreation system.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a
                href="/upload"
                className="inline-flex items-center bg-primary-gold hover:bg-yellow-500 text-primary-red font-sans font-bold py-3 px-8 rounded-full transition-all duration-300 transform hover:scale-105"
              >
                Start Creating
              </a>
              <a
                href="/gallery"
                className="inline-flex items-center border-2 border-white hover:bg-white hover:text-primary-red text-white font-sans font-bold py-3 px-8 rounded-full transition-all duration-300"
              >
                Explore Gallery
              </a>
            </div>
          </div>
        </motion.section>
      </div>
    </div>
  );
};

export default About;
